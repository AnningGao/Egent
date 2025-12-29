#!/usr/bin/env python3
"""
Local LLM Client
==============================

Runs Qwen3-VL locally using huggingface transformers.
No API key required - completely offline after model download.

Supported models:
- Qwen/Qwen3-VL-8B-Instruct
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class LocalLLMClient:
    """Local HuggingFace transformers client for EW measurement with Qwen3-VL."""
    
    # Default model (4-bit for memory efficiency on 16GB Macs)
    DEFAULT_MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
    
    def __init__(self, model_id: str = None):
        """
        Initialize local LLM client.
        
        Args:
            model_id: HuggingFace model ID.
                     Defaults to Qwen3-VL-8B-Instruct
        """
        self.model_id = model_id or self.DEFAULT_MODEL
        self._model = None
        self._processor = None
        self._loaded = False
        self._device = None
        self._device_cpu = 'cpu'

    def _ensure_loaded(self):
        """Lazy-load model on first use."""
        if self._loaded:
            return
            
        print(f"🔄 Loading local model: {self.model_id}")
        print("   (First run will download ~17GB from HuggingFace)")
        
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, 
                                                                          device_map="auto",
                                                                          attn_implementation='sdpa')
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._device = self._model.device
            print(f"   Model loaded on device: {self._device}")
            self._loaded = True
            print("✅ Model loaded successfully!")
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with:\n"
                "  pip install transformers\n"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def chat_with_vision(
        self,
        text_prompt: str,
        image_base64: str = None,
        image_path: str = None,
        timeout: int = 120,
        max_tokens: int = 1000,
    ) -> str:
        """
        Call local VLM with vision capability for plot inspection.
        
        Args:
            text_prompt: Text prompt
            image_base64: Base64-encoded image (optional, not recommended)
            image_path: Path to image file (recommended)
            timeout: Not used for local models
            max_tokens: Maximum tokens in response
            
        Returns:
            Response content as string
        """
        self._ensure_loaded()
        
        # Build message content
        content = []
        
        # Add image (prefer file path for better compatibility)
        if image_path:
            content.append({"type": "image", "image": str(image_path)})
        elif image_base64:
            # MLX-VLM has issues with base64, save to temp file
            import tempfile
            import base64
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(base64.b64decode(image_base64))
                temp_path = f.name
            content.append({"type": "image", "image": temp_path})
        
        # Add text
        content.append({"type": "text", "text": text_prompt})
        
        messages = [{"role": "user", "content": content}]
        print(messages)
        # Format prompt for model
        formatted = self._processor.apply_chat_template(
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            # config=self._model.config,
        )

        # Move inputs to device
        formatted = {k: v.to(self._device) for k, v in formatted.items()}

        # Generate response
        output_ids = self._model.generate(
            **formatted,
            max_new_tokens=max_tokens,
            # verbose=False
        )
        
        # Clean up temp file if created
        if image_base64 and not image_path:
            try:
                os.unlink(temp_path)
            except:
                pass
        
        output_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(formatted['input_ids'], output_ids)]
        response_text = self._processor.batch_decode([ids.to(self._device_cpu) for ids in output_ids_trimmed])[0]

        return response_text
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        system_prompt: str = None,
        timeout: int = 90,
        max_retries: int = 1,  # No retries needed for local
        initial_delay: float = 0,
    ) -> Any:
        """
        Chat with local LLM (with function calling support).
        
        Note: Function calling is emulated - Qwen3-VL doesn't have native
        tool use, so we parse the response for tool calls.
        
        Args:
            messages: List of message dicts
            tools: Tool definitions (for prompt construction)
            system_prompt: System prompt
            
        Returns:
            Response object mimicking OpenAI format
        """
        self._ensure_loaded()

        # Build the prompt
        full_messages = []
        
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        
        # Add tool descriptions to system prompt if tools provided
        if tools:
            tool_desc = self._format_tools_for_prompt(tools)
            if full_messages and full_messages[0]["role"] == "system":
                full_messages[0]["content"] += "\n\n" + tool_desc
            else:
                full_messages.insert(0, {"role": "system", "content": tool_desc})
        
        full_messages.extend(messages)
        
        # Check for images in messages
        has_image = False
        processed_messages = []
        for msg in full_messages:
            if isinstance(msg.get("content"), list):
                # Message with image
                content = []
                for item in msg["content"]:
                    if item.get("type") == "image_url":
                        # Convert OpenAI format to MLX format
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            # Base64 image - save to temp file
                            import base64
                            import tempfile
                            b64_data = url.split(",")[1]
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                                f.write(base64.b64decode(b64_data))
                                content.append({"type": "image", "image": f.name})
                        else:
                            content.append({"type": "image", "image": url})
                        has_image = True
                    elif item.get("type") == "text":
                        content.append(item)
                    else:
                        content.append(item)
                processed_messages.append({"role": msg["role"], "content": content})
            elif isinstance(msg.get("content"), str):
                # Text-only message
                message = {"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]}
                processed_messages.append(message)
        
        # Format for model
        formatted = self._processor.apply_chat_template(
            conversation=processed_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            # config=self._model.config,
        )
        # Move inputs to device
        formatted = {k: v.to(self._device) for k, v in formatted.items()}
        
        # Generate
        output_ids = self._model.generate(
            **formatted,
            max_new_tokens=2000,
            # verbose=False
        )

        # Parse response for tool calls
        output_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(formatted['input_ids'], output_ids)]
        output_ids_trimmed = [ids.to(self._device_cpu) for ids in output_ids_trimmed]
        response_text = self._processor.batch_decode(output_ids_trimmed)[0]
        tool_calls = self._parse_tool_calls(response_text, tools) if tools else None

        # Return OpenAI-compatible response object
        return LocalResponse(
            content=response_text,
            tool_calls=tool_calls,
            model=self.model_id,
            usage={
                "prompt_tokens": formatted['input_ids'].shape[1],
                "completion_tokens": output_ids_trimmed[0].shape[0],
                "total_tokens": output_ids[0].shape[0],
            }
        )
    
    def _format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """Format tools for prompt injection."""
        lines = ["You have access to the following tools:\n"]
        
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                lines.append(f"**{func['name']}**: {func.get('description', '')}")
                if func.get("parameters", {}).get("properties"):
                    lines.append("  Parameters:")
                    for name, prop in func["parameters"]["properties"].items():
                        lines.append(f"    - {name}: {prop.get('description', prop.get('type', 'any'))}")
                lines.append("")
        
        lines.append("\nTo use a tool, respond with a JSON block like:")
        lines.append('```json')
        lines.append('{"tool": "tool_name", "arguments": {"param1": "value1"}}')
        lines.append('```')
        lines.append("\nYou can call multiple tools by including multiple JSON blocks.")
        
        return "\n".join(lines)
    
    def _parse_tool_calls(self, text: str, tools: List[Dict]) -> Optional[List]:
        """Parse tool calls from response text."""
        import re
        
        # Look for JSON blocks
        pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if not matches:
            # Try inline JSON
            pattern = r'\{["\']tool["\']:\s*["\'][^"\']+["\'][^}]+\}'
            matches = re.findall(pattern, text)
        
        if not matches:
            return None
        
        tool_names = {t["function"]["name"] for t in tools if t.get("type") == "function"}
        tool_calls = []
        
        for i, match in enumerate(matches):
            try:
                data = json.loads(match)
                tool_name = data.get("tool") or data.get("name") or data.get("function")
                if tool_name in tool_names:
                    tool_calls.append(LocalToolCall(
                        id=f"call_{i}",
                        name=tool_name,
                        arguments=json.dumps(data.get("arguments", data.get("params", {})))
                    ))
            except json.JSONDecodeError:
                continue
        
        return tool_calls if tool_calls else None


class LocalResponse:
    """Mimics OpenAI response structure."""
    
    def __init__(self, content: str, tool_calls: list = None, model: str = "", usage: dict = None):
        self.choices = [LocalChoice(content, tool_calls)]
        self.model = model
        self.usage = usage or {}


class LocalChoice:
    """Mimics OpenAI choice structure."""
    
    def __init__(self, content: str, tool_calls: list = None):
        self.message = LocalMessage(content, tool_calls)
        self.finish_reason = "tool_calls" if tool_calls else "stop"


class LocalMessage:
    """Mimics OpenAI message structure."""
    
    def __init__(self, content: str, tool_calls: list = None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"


class LocalToolCall:
    """Mimics OpenAI tool call structure."""
    
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = LocalFunction(name, arguments)


class LocalFunction:
    """Mimics OpenAI function structure."""
    
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


# Quick test
if __name__ == "__main__":
    print("Testing LocalLLMClient...")
    client = LocalLLMClient()
    
    # Test text-only
    response = client.chat([{"role": "user", "content": "What is 2+2?"}])
    print(f"Response: {response.choices[0].message.content[:200]}...")

