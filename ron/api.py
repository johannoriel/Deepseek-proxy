from curl_cffi import requests
from typing import Optional, Dict, Any, Tuple, Literal
import json
from .pow import DeepSeekPOW
import sys
from pathlib import Path
import subprocess
import time

ThinkingMode = Literal['detailed', 'simple', 'disabled']
SearchMode = Literal['enabled', 'disabled']

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

        cookies_path = Path(__file__).parent / 'cookies.json'
        try:
            with open(cookies_path, 'r') as f:
                cookie_data = json.load(f)
                self.cookies = cookie_data.get('cookies', {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"\033[93mWarning: Could not load cookies from {cookies_path}: {e}\033[0m", file=sys.stderr)
            self.cookies = {}

    def _get_headers(self, pow_response: Optional[str] = None) -> Dict[str, str]:
        headers = {
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'authorization': f'Bearer {self.auth_token}',
            'content-type': 'application/json',
            'origin': 'https://chat.deepseek.com',
            'referer': 'https://chat.deepseek.com/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'x-app-version': '20241129.1',
            'x-client-locale': 'en_US',
            'x-client-platform': 'web',
            'x-client-version': '1.0.0-always',
        }
        if pow_response:
            headers['x-ds-pow-response'] = pow_response
        return headers

    def _refresh_cookies(self) -> None:
        try:
            script_path = Path(__file__).parent / 'bypass.py'
            subprocess.run([sys.executable, script_path], check=True)
            time.sleep(2)
            cookies_path = Path(__file__).parent / 'cookies.json'
            with open(cookies_path, 'r') as f:
                cookie_data = json.load(f)
                self.cookies = cookie_data.get('cookies', {})
        except Exception as e:
            print(f"\033[93mWarning: Failed to refresh cookies: {e}\033[0m", file=sys.stderr)

    def _make_request(self, method: str, endpoint: str, json_data: Dict[str, Any], pow_required: bool = False) -> Any:
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
                    impersonate='chrome120',
                    timeout=None
                )

                if "<!DOCTYPE html>" in response.text and "Just a moment" in response.text:
                    print("\033[93mWarning: Cloudflare protection detected. Bypassing...\033[0m", file=sys.stderr)
                    if retry_count < max_retries - 1:
                        self._refresh_cookies()
                        retry_count += 1
                        continue

                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                elif response.status_code >= 500:
                    raise APIError(f"Server error occurred: {response.text}", response.status_code)
                elif response.status_code != 200:
                    raise APIError(f"API request failed: {response.text}", response.status_code)

                return response.json()

            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Network error occurred: {str(e)}")
            except json.JSONDecodeError:
                raise APIError("Invalid JSON response from server")

        raise APIError("Failed to bypass Cloudflare protection after multiple attempts")

    def _get_pow_challenge(self) -> Dict[str, Any]:
        try:
            response = self._make_request(
                'POST',
                '/chat/create_pow_challenge',
                {'target_path': '/api/v0/chat/completion'}
            )
            return response['data']['biz_data']['challenge']
        except KeyError:
            raise APIError("Invalid challenge response format from server")

    def create_chat_session(self) -> str:
        """Creates a new chat session on DeepSeek's servers and returns the session ID."""
        try:
            response = self._make_request(
                'POST',
                '/chat_session/create',
                {'character_id': None}
            )
            return response['data']['biz_data']['id']
        except KeyError:
            raise APIError("Invalid session creation response format from server")

    def chat_completion(
        self,
        chat_session_id: str,
        prompt: str,
        parent_message_id: Optional[int] = None,
        thinking_enabled: bool = False,
        search_enabled: bool = False,
    ) -> Tuple[str, Optional[int]]:
        """
        Send a message and return (response_text, message_id).
        Pass the returned message_id as parent_message_id in the next call
        to correctly thread the conversation.
        """

        def consume_stream(response) -> Tuple[str, bool, Optional[int]]:
            text = ""
            incomplete = False
            message_id = None
            data_lines = []

            for raw in response.iter_lines():
                if not raw:
                    if data_lines:
                        joined = "\n".join(data_lines).strip()
                        try:
                            parsed = json.loads(joined)

                            if isinstance(parsed.get("v"), str):
                                text += parsed["v"]

                            elif isinstance(parsed.get("v"), dict):
                                resp = parsed["v"].get("response")
                                if resp and "message_id" in resp:
                                    message_id = resp["message_id"]

                            elif isinstance(parsed.get("v"), list):
                                for item in parsed["v"]:
                                    if item.get("p") == "status" and item.get("v") == "INCOMPLETE":
                                        incomplete = True

                        except Exception:
                            pass

                        data_lines = []
                    continue

                decoded = raw.decode(errors="ignore").strip()
                if decoded.startswith("data:"):
                    data_lines.append(decoded[5:].strip())

            return text, incomplete, message_id

        # First request
        payload = {
            "chat_session_id": chat_session_id,
            "parent_message_id": parent_message_id,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
        }

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
            timeout=60
        )

        full_text, incomplete, message_id = consume_stream(response)

        # Auto-resume if response was cut off
        while incomplete and message_id is not None:
            resume_payload = {
                "chat_session_id": chat_session_id,
                "message_id": message_id,
                "ack_to_resume": True
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
                timeout=60
            )

            resumed_text, incomplete, new_message_id = consume_stream(response)
            full_text += resumed_text

            if new_message_id is not None:
                message_id = new_message_id

        # Return both the text and the final message_id for threading
        return full_text, message_id
