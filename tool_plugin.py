# tool_plugin.py (updated - removed tool_placement)
"""
Plugin system for tool calling support in DeepSeek API.
Can be enabled/disabled via configuration.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
import time
import sys


def plugin_dbg(msg):
    """Debug function for plugin-specific messages"""
    print(f"[PLUGIN DEBUG] {msg}", file=sys.stderr, flush=True)


class ToolPlugin:
    """Base class for tool calling plugins."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        plugin_dbg(f"ToolPlugin base initialized, enabled={enabled}")

    def prepare_messages(
        self, messages: List[Dict], tools: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Modify messages before sending to DeepSeek API."""
        plugin_dbg(
            f"Base prepare_messages called with {len(messages)} messages, tools={tools is not None}"
        )
        return messages

    def process_response(
        self, response_text: str, original_messages: List[Dict]
    ) -> Tuple[str, Optional[List[Dict]]]:
        """Process the response text and extract tool calls if present."""
        plugin_dbg(
            f"Base process_response called with response_text length={len(response_text)}"
        )
        return response_text, None

    def prepare_tool_response(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_response: str,
        original_messages: List[Dict],
    ) -> Dict:
        """Prepare a tool response message to send back to the API."""
        plugin_dbg(f"Base prepare_tool_response called for tool {tool_name}")
        return {
            "role": "user",
            "content": f"[Tool Response for {tool_name}]: {tool_response}",
        }


class NativeToolPlugin(ToolPlugin):
    """Uses DeepSeek's native tool calling support."""

    def __init__(self, enabled: bool = False):
        super().__init__(enabled)
        plugin_dbg("NativeToolPlugin initialized")

    def prepare_messages(
        self, messages: List[Dict], tools: Optional[List[Dict]] = None
    ) -> List[Dict]:
        plugin_dbg(
            f"NativeToolPlugin.prepare_messages called, tools={tools is not None}"
        )
        # For native support, we don't modify messages
        return messages

    def process_response(
        self, response_text: str, original_messages: List[Dict]
    ) -> Tuple[str, Optional[List[Dict]]]:
        plugin_dbg("NativeToolPlugin.process_response called")
        # Native tool calls are already in the right format
        return response_text, None


class SimulatedToolPlugin(ToolPlugin):
    """
    Simulates tool calling through prompt engineering.
    Converts tool definitions to prompts and extracts tool calls from responses.
    """

    def __init__(self, enabled: bool = False):
        super().__init__(enabled)
        self.pending_tool_calls = {}  # Store tool calls between requests

        plugin_dbg("=" * 50)
        plugin_dbg("SimulatedToolPlugin INITIALIZED")
        plugin_dbg(f"Enabled: {enabled}")
        plugin_dbg("=" * 50)

    def _format_tools_prompt(self, tools: List[Dict]) -> str:
        """Convert tool definitions into a clear instruction prompt."""
        plugin_dbg(f"_format_tools_prompt called with {len(tools)} tools")

        prompt = "\n\nYou have access to the following tools/functions:\n\n"

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "unknown")
                description = func.get("description", "")
                parameters = func.get("parameters", {})

                plugin_dbg(f"  Formatting tool: {name}")

                prompt += f"Tool: {name}\n"
                prompt += f"Description: {description}\n"

                if parameters:
                    prompt += "Parameters:\n"
                    props = parameters.get("properties", {})
                    required = parameters.get("required", [])

                    for param_name, param_info in props.items():
                        req = "required" if param_name in required else "optional"
                        param_type = param_info.get("type", "string")
                        param_desc = param_info.get("description", "")
                        prompt += (
                            f"  - {param_name} ({param_type}, {req}): {param_desc}\n"
                        )
            prompt += "\n"

        prompt += """To use a tool, respond with a JSON object in the following format:
{
  "tool_calls": [
    {
      "name": "tool_name",
      "arguments": {
        "param1": "value1",
        "param2": "value2"
      }
    }
  ]
}

You may return the JSON directly, or wrapped in a markdown code block (```json ... ```). Do not add any other text before or after the JSON object when calling a tool.
"""

        plugin_dbg(f"Generated prompt length: {len(prompt)} characters")
        plugin_dbg(f"Prompt preview: {prompt[:200]}...")
        return prompt

    def prepare_messages(
        self, messages: List[Dict], tools: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Add tool definitions to system message."""
        plugin_dbg("=" * 50)
        plugin_dbg("SimulatedToolPlugin.prepare_messages CALLED")
        plugin_dbg(f"Input messages: {len(messages)}")
        plugin_dbg(f"Tools provided: {tools is not None}")

        if not tools:
            plugin_dbg("No tools provided, returning original messages")
            return messages

        plugin_dbg(f"Tools count: {len(tools)}")
        plugin_dbg(f"Tools data: {json.dumps(tools, indent=2)}")

        # Format tools prompt
        tools_prompt = self._format_tools_prompt(tools)

        # Always add to system message (original approach)
        return self._add_to_system_message(messages, tools_prompt)

    def _add_to_system_message(
        self, messages: List[Dict], tools_prompt: str
    ) -> List[Dict]:
        """Add tools prompt to system message."""
        plugin_dbg("Adding tools to system message")

        modified_messages = []
        system_message_found = False

        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                plugin_dbg(f"Found system message at index {i}")
                original_content = msg.get("content", "")
                plugin_dbg(f"Original system content: {original_content[:100]}...")

                modified_messages.append(
                    {
                        "role": "system",
                        "content": original_content + "\n\n" + tools_prompt,
                    }
                )
                system_message_found = True
                plugin_dbg("Added tools to existing system message")
            else:
                modified_messages.append(msg)

        if not system_message_found:
            plugin_dbg("No system message found, creating new one with tools")
            modified_messages.insert(0, {"role": "system", "content": tools_prompt})

        return modified_messages

    def _extract_json_blocks(self, text: str) -> List[Tuple[str, str]]:
        """Extract JSON blocks that may contain tool calls, returning list of (raw_text, json_str)."""
        found = []

        # Strategy 1: Try parsing entire response as JSON
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                data = json.loads(stripped)
                if self._is_tool_call_data(data):
                    plugin_dbg("Strategy 1: Entire response is JSON tool call")
                    return [(stripped, stripped)]
            except json.JSONDecodeError:
                pass

        # Strategy 2: Find JSON inside markdown code blocks (with or without language tag)
        # Matches ```json\n{...}\n``` or ```\n{...}\n```
        code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            block = match.group(1).strip()
            if block.startswith("{") or block.startswith("["):
                try:
                    data = json.loads(block)
                    if self._is_tool_call_data(data):
                        plugin_dbg(
                            f"Strategy 2: Found tool call in markdown code block"
                        )
                        found.append((match.group(0), block))
                        continue
                except json.JSONDecodeError:
                    pass

        if found:
            return found

        # Strategy 3: Find standalone JSON objects with tool_call patterns embedded in text
        # Match either full tool_calls wrapper or individual {name, arguments} objects
        json_patterns = [
            r'\{\s*"tool_calls"\s*:\s*\[.*?\]\s*\}',
            r'\{\s*"name"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}',
        ]

        for pattern in json_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                candidate = match.group(0)
                try:
                    data = json.loads(candidate)
                    if self._is_tool_call_data(data):
                        plugin_dbg(f"Strategy 3: Found embedded tool call via regex")
                        found.append((candidate, candidate))
                        break
                except json.JSONDecodeError:
                    pass
            if found:
                break

        return found

    def _is_tool_call_data(self, data: Any) -> bool:
        """Check if parsed JSON data contains tool call structure."""
        if isinstance(data, dict):
            if (
                "tool_calls" in data
                and isinstance(data["tool_calls"], list)
                and len(data["tool_calls"]) > 0
            ):
                return True
            if "name" in data and "arguments" in data:
                return True
        return False

    def _convert_to_openai_format(self, data: Dict) -> List[Dict]:
        """Convert extracted JSON data to OpenAI tool_call format."""
        tool_calls = []
        items = []

        if "tool_calls" in data:
            items = data["tool_calls"]
        elif "name" in data:
            items = [data]

        for tc in items:
            arguments = tc.get("arguments", {})
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)

            tool_call = {
                "id": tc.get("id", f"call_{len(tool_calls)}_{int(time.time() * 1000)}"),
                "type": "function",
                "function": {
                    "name": tc.get("name", tc.get("function", {}).get("name", "")),
                    "arguments": arguments,
                },
            }
            tool_calls.append(tool_call)

        return tool_calls

    def process_response(
        self, response_text: str, original_messages: List[Dict]
    ) -> Tuple[str, Optional[List[Dict]]]:
        """Extract tool calls from response text."""
        plugin_dbg("=" * 50)
        plugin_dbg("SimulatedToolPlugin.process_response CALLED")
        plugin_dbg(f"Response text length: {len(response_text)}")
        plugin_dbg(f"Response preview: {response_text[:200]}")

        tool_calls = []
        cleaned_text = response_text

        blocks = self._extract_json_blocks(response_text)
        plugin_dbg(f"Found {len(blocks)} JSON block(s) containing tool calls")

        for raw_text, json_str in blocks:
            try:
                data = json.loads(json_str)
                extracted = self._convert_to_openai_format(data)
                tool_calls.extend(extracted)
                cleaned_text = cleaned_text.replace(raw_text, "")
            except json.JSONDecodeError as e:
                plugin_dbg(f"Failed to parse JSON block: {e}")
                continue

        plugin_dbg(f"Total tool calls extracted: {len(tool_calls)}")

        if tool_calls:
            timestamp = str(time.time())
            self.pending_tool_calls[timestamp] = {
                "tool_calls": tool_calls,
                "timestamp": time.time(),
            }
            plugin_dbg(f"Stored {len(tool_calls)} tool calls with key {timestamp}")
            plugin_dbg(
                f"Tool calls in OpenAI format: {json.dumps(tool_calls, indent=2)}"
            )

        cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text).strip()
        plugin_dbg(f"Cleaned text length: {len(cleaned_text)}")
        plugin_dbg(f"Returning {len(tool_calls)} tool calls")
        plugin_dbg("=" * 50)

        return cleaned_text, tool_calls if tool_calls else None

    def prepare_tool_response(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_response: str,
        original_messages: List[Dict],
    ) -> Dict:
        """Convert tool response to a user message that the model can understand."""
        plugin_dbg("=" * 50)
        plugin_dbg("SimulatedToolPlugin.prepare_tool_response CALLED")
        plugin_dbg(f"Tool call ID: {tool_call_id}")
        plugin_dbg(f"Tool name: {tool_name}")
        plugin_dbg(f"Tool response length: {len(tool_response)}")
        plugin_dbg(f"Tool response preview: {tool_response[:200]}")

        result = {
            "role": "user",
            "content": f"The result of calling {tool_name} (ID: {tool_call_id}) is:\n\n{tool_response}\n\nPlease continue with your response based on this result.",
        }

        plugin_dbg(f"Created response message: {result['content'][:100]}...")
        plugin_dbg("=" * 50)
        return result

    def should_expect_tool_response(self, messages: List[Dict]) -> bool:
        """Check if we're in the middle of a tool conversation."""
        plugin_dbg("SimulatedToolPlugin.should_expect_tool_response CALLED")
        # Look for assistant messages with tool calls in the recent history
        for msg in reversed(messages[-10:]):  # Check last 10 messages
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                plugin_dbg(
                    f"Found assistant message with tool calls, expecting response"
                )
                return True
        plugin_dbg("No pending tool calls found")
        return False


# Plugin factory
def create_tool_plugin(
    plugin_type: str = "simulated", enabled: bool = False
) -> ToolPlugin:
    """Factory function to create the appropriate tool plugin."""
    plugin_dbg(f"create_tool_plugin called with type={plugin_type}, enabled={enabled}")

    if not enabled:
        plugin_dbg("Plugin disabled, returning base plugin")
        return ToolPlugin(enabled=False)

    if plugin_type == "native":
        plugin_dbg("Creating NativeToolPlugin")
        return NativeToolPlugin(enabled=True)
    elif plugin_type == "simulated":
        plugin_dbg("Creating SimulatedToolPlugin")
        return SimulatedToolPlugin(enabled=True)
    else:
        plugin_dbg(f"Unknown plugin type: {plugin_type}")
        raise ValueError(f"Unknown plugin type: {plugin_type}")
