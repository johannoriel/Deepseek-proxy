from curl_cffi import requests
from typing import Optional, Dict, Any, Tuple, Literal, List, Union
import json
from .pow import DeepSeekPOW
import sys
from pathlib import Path
import subprocess
import time

ThinkingMode = Literal["detailed", "simple", "disabled"]
SearchMode = Literal["enabled", "disabled"]


class DeepSeekError(Exception):
    pass


class AuthenticationError(DeepSeekError):
    pass


class RateLimitError(DeepSeekError):
    pass


class NetworkError(DeepSeekError):
    pass


class CloudflareError(DeepSeekError):
    pass


class APIError(DeepSeekError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class DeepSeekAPI:
    BASE_URL = "https://chat.deepseek.com/api/v0"

    def __init__(self, auth_token: str):
        if not auth_token or not isinstance(auth_token, str):
            raise AuthenticationError("Invalid auth token provided")

        self.auth_token = auth_token
        self.pow_solver = DeepSeekPOW()

        cookies_path = Path(__file__).parent / "cookies.json"
        try:
            with open(cookies_path, "r") as f:
                cookie_data = json.load(f)
                self.cookies = cookie_data.get("cookies", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(
                f"\033[93mWarning: Could not load cookies from {cookies_path}: {e}\033[0m",
                file=sys.stderr,
            )
            self.cookies = {}

    def _get_headers(self, pow_response: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "accept": "*/*",
            "accept-language": "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "authorization": f"Bearer {self.auth_token}",
            "content-type": "application/json",
            "origin": "https://chat.deepseek.com",
            "referer": "https://chat.deepseek.com/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
            "x-app-version": "20241129.1",
            "x-client-locale": "en_US",
            "x-client-platform": "web",
            "x-client-version": "1.0.0-always",
        }
        if pow_response:
            headers["x-ds-pow-response"] = pow_response
        return headers

    def _refresh_cookies(self) -> None:
        try:
            script_path = Path(__file__).parent / "bypass.py"
            subprocess.run([sys.executable, script_path], check=True)
            time.sleep(2)
            cookies_path = Path(__file__).parent / "cookies.json"
            with open(cookies_path, "r") as f:
                cookie_data = json.load(f)
                self.cookies = cookie_data.get("cookies", {})
        except Exception as e:
            print(
                f"\033[93mWarning: Failed to refresh cookies: {e}\033[0m",
                file=sys.stderr,
            )

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Dict[str, Any],
        pow_required: bool = False,
    ) -> Any:
        url = f"{self.BASE_URL}{endpoint}"
        retry_count = 0
        max_retries = 2

        while retry_count < max_retries:
            try:
                headers = self._get_headers()
                if pow_required:
                    challenge = self._get_pow_challenge()
                    pow_response = self.pow_solver.solve_challenge(challenge)
                    headers = self._get_headers(pow_response)

                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data,
                    cookies=self.cookies,
                    impersonate="chrome120",
                    timeout=None,
                )

                # Check for HTML/Cloudflare response
                if (
                    "<!DOCTYPE html>" in response.text
                    and "Just a moment" in response.text
                ):
                    print(
                        "\033[93mWarning: Cloudflare protection detected. Bypassing...\033[0m",
                        file=sys.stderr,
                    )
                    if retry_count < max_retries - 1:
                        self._refresh_cookies()
                        retry_count += 1
                        continue
                    else:
                        raise CloudflareError(
                            "Failed to bypass Cloudflare protection after all retries"
                        )

                # Check status codes
                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                elif response.status_code >= 500:
                    raise APIError(
                        f"Server error occurred: {response.text}", response.status_code
                    )
                elif response.status_code != 200:
                    raise APIError(
                        f"API request failed with status {response.status_code}: {response.text}",
                        response.status_code,
                    )

                # Parse JSON response
                try:
                    json_response = response.json()
                except json.JSONDecodeError as e:
                    raise APIError(
                        f"Invalid JSON response from server (first 200 chars): {response.text[:200]}"
                    )

                # Log the response for debugging (optional)
                dbg(
                    f"API Response for {endpoint}: status={response.status_code}, has_data={bool(json_response.get('data'))}"
                )

                return json_response

            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Network error occurred: {str(e)}")
            except (AuthenticationError, RateLimitError, APIError, CloudflareError):
                raise
            except Exception as e:
                raise APIError(f"Unexpected error in request: {str(e)}")

        raise APIError("Failed to bypass Cloudflare protection after multiple attempts")

    def _get_pow_challenge(self) -> Dict[str, Any]:
        try:
            response = self._make_request(
                "POST",
                "/chat/create_pow_challenge",
                {"target_path": "/api/v0/chat/completion"},
            )
            return response["data"]["biz_data"]["challenge"]
        except KeyError:
            raise APIError("Invalid challenge response format from server")

    def create_chat_session(self) -> str:
        """Creates a new chat session on DeepSeek's servers and returns the session ID."""
        try:
            response = self._make_request(
                "POST", "/chat_session/create", {"character_id": None}
            )

            # Add detailed error checking
            if not response:
                raise APIError(
                    "Empty response received from server when creating chat session"
                )

            if not isinstance(response, dict):
                raise APIError(
                    f"Unexpected response type: {type(response)}. Expected dict"
                )

            if "data" not in response:
                raise APIError(f"Missing 'data' field in response: {response}")

            if response["data"] is None:
                raise APIError(
                    f"'data' field is None in response. Full response: {response}"
                )

            if "biz_data" not in response["data"]:
                raise APIError(
                    f"Missing 'biz_data' field in response['data']: {response['data']}"
                )

            if "id" not in response["data"]["biz_data"]:
                raise APIError(
                    f"Missing 'id' field in response['data']['biz_data']: {response['data']['biz_data']}"
                )

            return response["data"]["biz_data"]["id"]

        except KeyError as e:
            raise APIError(
                f"Invalid session creation response format from server: missing key {e}. Full response: {response if 'response' in locals() else 'No response'}"
            )
        except Exception as e:
            raise APIError(f"Failed to create chat session: {str(e)}")

    def create_chat_session_with_retry(self, max_retries: int = 3) -> str:
        """Creates a new chat session with retry logic."""
        for attempt in range(max_retries):
            try:
                return self.create_chat_session()
            except APIError as e:
                if attempt == max_retries - 1:
                    raise
                dbg(
                    f"Session creation failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(2**attempt)  # Exponential backoff
        raise APIError("Failed to create chat session after all retries")

    def _parse_tool_calls_from_stream(
        self, data_lines: List[str]
    ) -> Tuple[Optional[List[Dict]], str]:
        """Parse tool calls from streaming response data."""
        tool_calls = []
        text_parts = []

        for line in data_lines:
            try:
                parsed = json.loads(line)
                if isinstance(parsed.get("v"), dict):
                    # Check for tool calls in the response
                    if "tool_calls" in parsed["v"]:
                        for tc in parsed["v"]["tool_calls"]:
                            tool_calls.append(
                                {
                                    "id": tc.get(
                                        "id",
                                        f"call_{len(tool_calls)}_{int(time.time())}",
                                    ),
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("function", {}).get("name", ""),
                                        "arguments": tc.get("function", {}).get(
                                            "arguments", "{}"
                                        ),
                                    },
                                }
                            )
                    # Get text content
                    if "content" in parsed["v"]:
                        text_parts.append(parsed["v"]["content"])
                elif isinstance(parsed.get("v"), str):
                    text_parts.append(parsed["v"])
            except Exception:
                # If parsing fails, treat as text
                text_parts.append(line)

        return tool_calls if tool_calls else None, "".join(text_parts)

    def chat_completion(
        self,
        chat_session_id: str,
        prompt: str,
        parent_message_id: Optional[int] = None,
        thinking_enabled: bool = False,
        search_enabled: bool = False,
        tools: Optional[List[Dict]] = None,
        tool_choice: Union[str, Dict] = "auto",
    ) -> Union[Tuple[str, Optional[int]], Dict[str, Any]]:
        """
        Send a message and return response.
        If tools are provided and the model decides to use them, returns a dict with tool_calls.
        Otherwise returns (response_text, message_id).
        """

        def consume_stream(
            response,
        ) -> Tuple[str, bool, Optional[int], Optional[List[Dict]]]:
            text = ""
            incomplete = False
            message_id = None
            tool_calls = None
            line_count = 0
            data_count = 0
            error_toast = None

            for raw in response.iter_lines():
                if not raw:
                    continue

                decoded = raw.decode(errors="ignore").strip()
                line_count += 1

                if decoded.startswith("event:"):
                    continue

                if decoded.startswith("data:"):
                    data_str = decoded[5:].strip()
                    data_count += 1

                    try:
                        parsed = json.loads(data_str)

                        if isinstance(parsed.get("v"), str):
                            text += parsed["v"]

                        elif isinstance(parsed.get("v"), dict):
                            resp = parsed["v"].get("response")
                            if resp and "message_id" in resp:
                                message_id = resp["message_id"]

                            if "tool_calls" in parsed["v"]:
                                tc_data = parsed["v"]["tool_calls"]
                                if tc_data and not tool_calls:
                                    tool_calls = []
                                    for i, tc in enumerate(tc_data):
                                        tool_calls.append(
                                            {
                                                "id": tc.get(
                                                    "id", f"call_{i}_{int(time.time())}"
                                                ),
                                                "type": "function",
                                                "function": {
                                                    "name": tc.get("function", {}).get(
                                                        "name", ""
                                                    ),
                                                    "arguments": json.dumps(
                                                        tc.get("function", {}).get(
                                                            "arguments", {}
                                                        )
                                                    ),
                                                },
                                            }
                                        )

                        elif isinstance(parsed.get("v"), list):
                            for item in parsed["v"]:
                                if (
                                    item.get("p") == "status"
                                    and item.get("v") == "INCOMPLETE"
                                ):
                                    incomplete = True

                        # Detect error toast messages (no "v" key, has "type":"error")
                        if parsed.get("type") == "error":
                            error_toast = parsed

                    except Exception as e:
                        dbg(f"  STREAM PARSE ERROR: {e}, data: {data_str[:200]}")

            if error_toast:
                finish_reason = error_toast.get("finish_reason", "unknown")
                error_content = error_toast.get("content", "Unknown error")
                raise APIError(
                    f"DeepSeek stream error ({finish_reason}): {error_content}",
                    status_code=429 if finish_reason == "rate_limit_reached" else 500,
                )

            dbg(
                f"  STREAM SUMMARY: {line_count} raw lines, {data_count} data lines, text_len={len(text)}, message_id={message_id}"
            )
            return text, incomplete, message_id, tool_calls

        # Build the payload with all supported parameters
        payload = {
            "chat_session_id": chat_session_id,
            "parent_message_id": parent_message_id,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
        }

        # Add tools to payload if provided (native DeepSeek API support)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        dbg(f"chat_completion: Sending payload with tools={tools is not None}")

        max_retries = 10
        retry_delay = 6
        total_waited = 0

        for attempt in range(max_retries):
            challenge = self._get_pow_challenge()
            pow_response = self.pow_solver.solve_challenge(challenge)
            headers = self._get_headers(pow_response)

            response = requests.post(
                f"{self.BASE_URL}/chat/completion",
                headers=headers,
                json=payload,
                cookies=self.cookies,
                impersonate="chrome120",
                stream=True,
                timeout=60,
            )

            try:
                full_text, incomplete, message_id, tool_calls = consume_stream(response)
                if total_waited > 0:
                    dbg(
                        f"chat_completion: SUCCEEDED after {total_waited}s total wait across {attempt} retry(ies)"
                    )
                break
            except APIError as e:
                if e.status_code == 429 and attempt < max_retries - 1:
                    dbg(
                        f"chat_completion: Rate limited, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                    total_waited += retry_delay
                    retry_delay = retry_delay * 2
                    continue
                raise

        # Auto-resume if response was cut off
        while incomplete and message_id is not None:
            resume_payload = {
                "chat_session_id": chat_session_id,
                "message_id": message_id,
                "ack_to_resume": True,
            }

            challenge = self._get_pow_challenge()
            pow_response = self.pow_solver.solve_challenge(challenge)
            headers = self._get_headers(pow_response)

            response = requests.post(
                f"{self.BASE_URL}/chat/continue",
                headers=headers,
                json=resume_payload,
                cookies=self.cookies,
                impersonate="chrome120",
                stream=True,
                timeout=60,
            )

            resumed_text, incomplete, new_message_id, new_tool_calls = consume_stream(
                response
            )
            full_text += resumed_text

            # Merge tool calls if any
            if new_tool_calls and not tool_calls:
                tool_calls = new_tool_calls

            if new_message_id is not None:
                message_id = new_message_id

        # If we have tool calls, return them in a structured format
        if tool_calls:
            dbg(f"chat_completion: Detected {len(tool_calls)} tool calls in response")
            return {
                "content": full_text,
                "tool_calls": tool_calls,
                "message_id": message_id,
            }
        else:
            # Regular text response
            return full_text, message_id


# Helper function for debug logging
def dbg(msg):
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)
