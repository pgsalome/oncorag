"""LLM adapter system for different backends."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import ollama
from ..config.system_config import LLMBackend, LLMConfig, get_system_config
from ..utils.phi_removal import remove_phi_from_context, should_remove_phi
from ..utils.logging_utils import log


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def query(self, prompt: str, context: str) -> Dict[str, Any]:
        """Query the LLM with prompt and context."""
        pass
    
    def _prepare_context(self, context: str) -> str:
        """Prepare context for LLM, including PHI removal if needed."""
        if should_remove_phi():
            log("Removing PHI from context before LLM processing...", level="STEP")
            phi_result = remove_phi_from_context(context)
            
            if phi_result.removed_count > 0:
                log(f"Removed {phi_result.removed_count} PHI elements: {', '.join(phi_result.removed_types)}", 
                    level="WARNING")
                if self.config.phi_removal_patterns:
                    log(f"PHI patterns used: {', '.join(self.config.phi_removal_patterns)}", 
                        level="INFO", debug=True)
            
            return phi_result.cleaned_text
        else:
            log("PHI removal not required for current backend", level="INFO", debug=True)
            return context


class OllamaAdapter(LLMAdapter):
    """Adapter for local Ollama models."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = ollama.Client(
            host=os.getenv("OLLAMA_HOST") or config.base_url or "http://127.0.0.1:11434",
            timeout=config.timeout,
        )

    def _generation_options(self) -> dict:
        options = dict(self.config.extra_params or {})
        options["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            options["num_predict"] = self.config.max_tokens
        if self.config.num_ctx is not None:
            options["num_ctx"] = self.config.num_ctx
        return options

    @staticmethod
    def _decode_response(text: str) -> Dict[str, Any]:
        def reject_constant(value):
            raise ValueError(f"Invalid JSON numeric constant: {value}")

        response = json.loads(text, parse_constant=reject_constant)
        if not isinstance(response, dict):
            raise ValueError("Ollama response must be a JSON object")
        return response

    @staticmethod
    def _json_max_tries() -> int:
        """Max parse attempts per Ollama response cycle (including first attempt)."""
        raw = os.getenv("ONCORAG_LLM_JSON_MAX_TRIES", "6")
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 6
    
    def query(self, prompt: str, context: str) -> Dict[str, Any]:
        """Query Ollama with the given prompt and context."""
        # Check for OLLAMA_MODEL environment variable override
        model_name = os.getenv("OLLAMA_MODEL") or self.config.model_name
        log(f"Querying Ollama ({model_name}) for extraction...", level="STEP")
        
        # Prepare context (with PHI removal if needed)
        prepared_context = self._prepare_context(context)
        
        # Update prompt with prepared context
        full_prompt = prompt.replace(context, prepared_context)
        
        try:
            response = self.client.chat(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                format="json",
                options=self._generation_options(),
            )
            
            raw_response = response["message"]["content"]
            return self._parse_response(raw_response, model_name=model_name, original_prompt=full_prompt)
            
        except Exception as exc:
            log(f"Ollama error: {exc}", level="ERROR")
            return self._error_response(str(exc))
    
    def _parse_response(self, raw_response: str, model_name: str | None = None, original_prompt: str | None = None) -> Dict[str, Any]:
        """Parse Ollama response."""
        try:
            return self._decode_response(raw_response)
        except (ValueError, TypeError):
            retry_model = model_name or os.getenv("OLLAMA_MODEL") or self.config.model_name
            retry_prompt = raw_response + "\n\nReturn ONLY strict JSON (double quotes, no trailing commas)."
            retry_messages = [{"role": "user", "content": retry_prompt}]
            if original_prompt is not None:
                retry_messages = [
                    {"role": "user", "content": original_prompt},
                    {"role": "assistant", "content": raw_response},
                    {"role": "user", "content": "Return ONLY a strict JSON object for the original extraction request."},
                ]
            max_tries = self._json_max_tries()
            attempt = 2
            while attempt <= max_tries:
                log(
                    f"Retrying with stricter JSON format... (attempt {attempt}/{max_tries})",
                    level="WARNING",
                )
                try:
                    response = self.client.chat(
                        model=retry_model,
                        messages=retry_messages,
                        format="json",
                        options=self._generation_options(),
                    )
                    repaired = response["message"]["content"]
                    return self._decode_response(repaired)
                except (ValueError, TypeError):
                    attempt += 1
                    continue
                except Exception:
                    attempt += 1
                    continue
            return self._error_response("Failed to parse JSON response after retries")
    
    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Return error response."""
        return {
            # Use Missing so categorical configs map cleanly to the Missing option.
            "value": "Missing",
            "reasoning": f"error_during_extraction: {error_msg}",
            "confidence": "Low",
        }


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._validate_api_key()
    
    def _validate_api_key(self):
        """Validate OpenAI API key."""
        api_key = get_system_config().get_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    def query(self, prompt: str, context: str) -> Dict[str, Any]:
        """Query OpenAI API with the given prompt and context."""
        log(f"Querying OpenAI ({self.config.model_name}) for extraction...", level="STEP")
        
        # Prepare context (with PHI removal)
        prepared_context = self._prepare_context(context)
        
        # Update prompt with prepared context
        full_prompt = prompt.replace(context, prepared_context)
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=get_system_config().get_api_key())
            
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
            )
            
            raw_response = response.choices[0].message.content
            return self._parse_response(raw_response)
            
        except Exception as exc:
            log(f"OpenAI API error: {exc}", level="ERROR")
            return self._error_response(str(exc))
    
    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse OpenAI response."""
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            return self._error_response("Failed to parse JSON response")
    
    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Return error response."""
        return {
            "value": "error_during_extraction",
            "reasoning": error_msg,
            "confidence": "Low",
        }


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic API."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._validate_api_key()
    
    def _validate_api_key(self):
        """Validate Anthropic API key."""
        api_key = get_system_config().get_api_key()
        if not api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
    
    def query(self, prompt: str, context: str) -> Dict[str, Any]:
        """Query Anthropic API with the given prompt and context."""
        log(f"Querying Anthropic ({self.config.model_name}) for extraction...", level="STEP")
        
        # Prepare context (with PHI removal)
        prepared_context = self._prepare_context(context)
        
        # Update prompt with prepared context
        full_prompt = prompt.replace(context, prepared_context)
        
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=get_system_config().get_api_key())
            
            response = client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens or 1000,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": full_prompt}],
            )
            
            raw_response = response.content[0].text
            return self._parse_response(raw_response)
            
        except Exception as exc:
            log(f"Anthropic API error: {exc}", level="ERROR")
            return self._error_response(str(exc))
    
    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse Anthropic response."""
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            return self._error_response("Failed to parse JSON response")
    
    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Return error response."""
        return {
            "value": "error_during_extraction",
            "reasoning": error_msg,
            "confidence": "Low",
        }


class LocalHuggingFaceAdapter(LLMAdapter):
    """Adapter for local HuggingFace models."""
    
    def query(self, prompt: str, context: str) -> Dict[str, Any]:
        """Query local HuggingFace model."""
        log(f"Querying local HuggingFace model ({self.config.model_name})...", level="STEP")
        
        # Prepare context (no PHI removal needed for local models)
        prepared_context = self._prepare_context(context)
        
        # Update prompt with prepared context
        full_prompt = prompt.replace(context, prepared_context)
        
        try:
            # This is a placeholder - you'd implement actual HuggingFace model loading here
            # For now, return a mock response
            log("Local HuggingFace adapter not fully implemented", level="WARNING")
            return {
                "value": "not_implemented",
                "reasoning": "Local HuggingFace adapter needs implementation",
                "confidence": "Low",
            }
            
        except Exception as exc:
            log(f"Local HuggingFace error: {exc}", level="ERROR")
            return self._error_response(str(exc))
    
    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Return error response."""
        return {
            "value": "error_during_extraction",
            "reasoning": error_msg,
            "confidence": "Low",
        }


def create_llm_adapter(config: LLMConfig) -> LLMAdapter:
    """Create appropriate LLM adapter based on configuration."""
    if config.backend == LLMBackend.OLLAMA_LOCAL:
        return OllamaAdapter(config)
    elif config.backend == LLMBackend.OPENAI_API:
        return OpenAIAdapter(config)
    elif config.backend == LLMBackend.ANTHROPIC_API:
        return AnthropicAdapter(config)
    elif config.backend == LLMBackend.AZURE_OPENAI:
        # Azure OpenAI uses similar interface to OpenAI
        return OpenAIAdapter(config)
    elif config.backend == LLMBackend.LOCAL_HF:
        return LocalHuggingFaceAdapter(config)
    else:
        raise ValueError(f"Unsupported LLM backend: {config.backend}")


def get_llm_adapter() -> LLMAdapter:
    """Get LLM adapter for current system configuration."""
    config = get_system_config()
    llm_config = config.get_llm_config()
    return create_llm_adapter(llm_config)


__all__ = [
    "LLMAdapter",
    "OllamaAdapter", 
    "OpenAIAdapter",
    "AnthropicAdapter",
    "LocalHuggingFaceAdapter",
    "create_llm_adapter",
    "get_llm_adapter"
]
