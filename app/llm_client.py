"""
Unified LLM client that supports both Ollama and OpenAI providers.
"""
from __future__ import annotations
import os
import json
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

class LLMClient:
    """Unified client for both Ollama and OpenAI providers."""
    
    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the LLM."""
        if self.provider == "ollama":
            return self._generate_ollama(messages, **kwargs)
        elif self.provider == "openai":
            return self._generate_openai(messages, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_ollama(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate using Ollama."""
        # Convert messages to a single prompt
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.2),
                "max_tokens": kwargs.get("max_tokens", 500)
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")
    
    def _generate_openai(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate using OpenAI."""
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 500)
        )
        return response.choices[0].message.content.strip()
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to a single prompt for Ollama."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

def get_client(provider_key: str, model_key: str) -> LLMClient:
    """Get an LLM client based on environment variables."""
    provider = os.getenv(provider_key, "openai")
    model = os.getenv(model_key, "gpt-4o-mini")
    return LLMClient(provider, model)
