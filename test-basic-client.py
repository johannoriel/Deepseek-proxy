#!/usr/bin/env python3
"""
Simple OpenAI-compatible client with tool/function calling.
Example: List directory contents using a tool.
"""

import os
import json
import yaml
import subprocess
from openai import OpenAI
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def list_directory_tool(directory: str = ".") -> str:
    """
    Tool function that lists directory contents.
    This is the actual implementation of the tool.
    """
    try:
        # List directory contents
        files = os.listdir(directory)
        
        # Get additional info for files
        result = []
        for f in sorted(files):
            full_path = os.path.join(directory, f)
            if os.path.isdir(full_path):
                result.append(f"{f}/")
            else:
                size = os.path.getsize(full_path)
                result.append(f"{f} ({size} bytes)")
        
        return "\n".join(result) if result else "(empty directory)"
    except Exception as e:
        return f"Error listing directory: {str(e)}"

def main():
    # Load configuration
    config = load_config()
    
    # Setup OpenAI client for local API
    base_url = f"{config['api']['url']}:{config['api']['port']}/v1"
    client = OpenAI(
        base_url=base_url,
        api_key=config.get('api_key', 'dummy-key'),  # Some APIs need a key
    )
    
    # Define the tool/function
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List contents of a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Path to the directory to list (default: current directory)",
                            "default": "."
                        }
                    },
                    "additionalProperties": False
                }
            }
        }
    ]
    
    # Initial user message that should trigger tool use
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can list directories."},
        {"role": "user", "content": "What files are in the current directory?"}
    ]
    
    print(f"🤖 Sending request to {base_url} with model: {config['api']['model']}")
    print("📝 User query:", messages[-1]["content"])
    print("-" * 50)
    
    # First API call - should respond with tool call
    response = client.chat.completions.create(
        model=config['api']['model'],
        messages=messages,
        tools=tools,
        tool_choice="auto"  # Let model decide, but it will likely use the tool
    )
    
    # Check if model wants to use a tool
    message = response.choices[0].message
    
    if message.tool_calls:
        # Handle tool calls
        for tool_call in message.tool_calls:
            if tool_call.function.name == "list_directory":
                print(f"🛠️  Tool called: {tool_call.function.name}")
                
                # Parse arguments
                args = json.loads(tool_call.function.arguments)
                directory = args.get("directory", ".")
                print(f"📂 Listing directory: {directory}")
                
                # Execute the tool
                tool_result = list_directory_tool(directory)
                print(f"📋 Result:\n{tool_result}")
                
                # Add the assistant's message with tool calls
                messages.append(message)
                
                # Add tool response message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
                
                print("-" * 50)
                print("🔄 Sending tool result back to LLM...")
                
                # Second API call - get final response
                final_response = client.chat.completions.create(
                    model=config['api']['model'],
                    messages=messages,
                    tools=tools,
                )
                
                # Print final response
                final_message = final_response.choices[0].message
                print(f"🤖 Final response: {final_message.content}")
    else:
        # Model didn't use tool, just print response
        print(f"🤖 Response: {message.content}")

if __name__ == "__main__":
    main()
