"""System configuration for LLM backends and PHI handling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class LLMBackend(Enum):
    """Supported LLM backends."""

    OLLAMA_LOCAL = "ollama_local"
    OPENAI_API = "openai_api"
    ANTHROPIC_API = "anthropic_api"
    AZURE_OPENAI = "azure_openai"
    LOCAL_HF = "local_huggingface"


@dataclass
class LLMConfig:
    """Configuration for a specific LLM backend."""
    backend: LLMBackend
    model_name: str
    requires_phi_removal: bool
    api_key_env_var: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    num_ctx: Optional[int] = None
    timeout: int = 120
    phi_removal_patterns: Optional[list[str]] = None
    phi_replacement: str = "[PHI_REDACTED]"
    extra_params: Optional[Dict[str, Any]] = None


class SystemConfig:
    """Main system configuration manager."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.current_llm_config = self._get_llm_config()

    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path."""
        package_root = Path(__file__).resolve().parents[1]
        return package_root / "system_config.yaml"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            default_config = self._get_default_config()
            self._save_config(default_config)
            return default_config

        with open(self.config_path, "r") as fh:
            return yaml.safe_load(fh) or {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Generate a default configuration."""
        return {
            "llm_backend": "ollama_local",
            "llm_configs": {
                "ollama_local": {
                    "backend": "ollama_local",
                    "model_name": "phi3:medium",
                    "requires_phi_removal": False,
                    "base_url": "http://localhost:11434",
                    "temperature": 0.1,
                    "num_ctx": None,
                    "timeout": 120,
                },
                "openai_api": {
                    "backend": "openai_api",
                    "model_name": "gpt-4o-mini",
                    "requires_phi_removal": True,
                    "api_key_env_var": "OPENAI_API_KEY",
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "timeout": 120,
                    "phi_removal_patterns": [
                        "ssn",
                        "phone",
                        "email",
                        "address",
                        "patient_name",
                        "mrn",
                        "dob",
                        "zip_code",
                        "credit_card",
                    ],
                },
                "anthropic_api": {
                    "backend": "anthropic_api",
                    "model_name": "claude-3-haiku-20240307",
                    "requires_phi_removal": True,
                    "api_key_env_var": "ANTHROPIC_API_KEY",
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "timeout": 120,
                    "phi_removal_patterns": [
                        "ssn",
                        "phone",
                        "email",
                        "address",
                        "patient_name",
                        "mrn",
                        "dob",
                        "zip_code",
                    ],
                },
                "azure_openai": {
                    "backend": "azure_openai",
                    "model_name": "gpt-4o-mini",
                    "requires_phi_removal": True,
                    "api_key_env_var": "AZURE_OPENAI_API_KEY",
                    "base_url": "https://your-resource.openai.azure.com/",
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "timeout": 120,
                    "phi_removal_patterns": [
                        "ssn",
                        "phone",
                        "email",
                        "address",
                        "patient_name",
                        "mrn",
                        "dob",
                        "zip_code",
                    ],
                    "extra_params": {"api_version": "2024-02-15-preview"},
                },
                "local_huggingface": {
                    "backend": "local_huggingface",
                    "model_name": "microsoft/DialoGPT-medium",
                    "requires_phi_removal": False,
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "num_ctx": None,
                    "timeout": 120,
                },
            },
            "phi_removal": {
                "patterns": {
                    "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
                    "phone": r"\b\d{3}-?\d{3}-?\d{4}\b",
                    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    "address": r"\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b",
                    "zip_code": r"\b\d{5}(?:-\d{4})?\b",
                    "patient_name": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
                    "mrn": r"\b(?:MRN|Medical Record Number):\s*\d+\b",
                    "dob": r"\b(?:DOB|Date of Birth):\s*\d{1,2}/\d{1,2}/\d{4}\b",
                    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                },
                "replacement": "[PHI_REDACTED]",
            },
            "logging": {
                "log_phi_removal": True,
                "log_llm_requests": False,
                "level": "INFO",
            },
        }

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Persist configuration to disk."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as fh:
            yaml.dump(config, fh, default_flow_style=False, indent=2)

    def _get_llm_config(self) -> LLMConfig:
        """Construct the current LLM configuration."""
        backend_name = self.config.get("llm_backend", "ollama_local")
        llm_config_data = self.config["llm_configs"][backend_name]
        num_ctx = llm_config_data.get("num_ctx")
        env_num_ctx = os.getenv("ONCORAG_OLLAMA_NUM_CTX")
        if env_num_ctx not in (None, ""):
            try:
                num_ctx = int(env_num_ctx)
            except ValueError:
                pass
        return LLMConfig(
            backend=LLMBackend(llm_config_data["backend"]),
            model_name=llm_config_data["model_name"],
            requires_phi_removal=llm_config_data["requires_phi_removal"],
            api_key_env_var=llm_config_data.get("api_key_env_var"),
            base_url=llm_config_data.get("base_url"),
            temperature=llm_config_data.get("temperature", 0.1),
            max_tokens=llm_config_data.get("max_tokens"),
            num_ctx=num_ctx,
            timeout=llm_config_data.get("timeout", 120),
            phi_removal_patterns=llm_config_data.get("phi_removal_patterns"),
            phi_replacement=llm_config_data.get("phi_replacement", "[PHI_REDACTED]"),
            extra_params=llm_config_data.get("extra_params"),
        )

    def get_llm_config(self) -> LLMConfig:
        return self.current_llm_config

    def set_llm_backend(self, backend_name: str) -> None:
        if backend_name not in self.config["llm_configs"]:
            raise ValueError(f"Unknown LLM backend: {backend_name}")
        self.config["llm_backend"] = backend_name
        self.current_llm_config = self._get_llm_config()
        self._save_config(self.config)

    def get_phi_removal_patterns(self) -> Dict[str, str]:
        return self.config["phi_removal"]["patterns"]

    def get_phi_replacement(self) -> str:
        return self.config["phi_removal"]["replacement"]

    def should_remove_phi(self) -> bool:
        return self.current_llm_config.requires_phi_removal

    def get_api_key(self) -> Optional[str]:
        if not self.current_llm_config.api_key_env_var:
            return None
        return os.getenv(self.current_llm_config.api_key_env_var)

    def validate_config(self) -> bool:
        try:
            if self.current_llm_config.api_key_env_var and not self.get_api_key():
                return False
            return True
        except Exception:
            return False


_system_config: Optional[SystemConfig] = None


def get_system_config() -> SystemConfig:
    global _system_config
    if _system_config is None:
        _system_config = SystemConfig()
    return _system_config


def set_llm_backend(backend_name: str) -> None:
    get_system_config().set_llm_backend(backend_name)


__all__ = [
    "LLMBackend",
    "LLMConfig",
    "SystemConfig",
    "get_system_config",
    "set_llm_backend",
]
