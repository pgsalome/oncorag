"""Model initialization and caching helpers with GPU-aware placement."""

from __future__ import annotations

import os
import subprocess
import warnings
from typing import Optional

import yaml
from pathlib import Path

import medspacy
from medspacy.context import ConTextRule
import spacy
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder, SentenceTransformer

# Suppress transformers warnings about uninitialized weights (expected for reranking models)
try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass

# Suppress sentence-transformers info messages about model creation
import logging
sentence_transformers_logger = logging.getLogger("sentence_transformers")
sentence_transformers_logger.setLevel(logging.WARNING)

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None

from ..utils.logging_utils import log

RERANKER_PRIMARY: Optional[CrossEncoder] = None
RERANKER_SECONDARY: Optional[CrossEncoder] = None
MODEL_CACHE: dict[str, spacy.language.Language] = {}
NLP_MED = None
CLINICAL_EMBEDDER: Optional[SentenceTransformer] = None
CHROMA_EMBEDDER: Optional[SentenceTransformer] = None
CHROMA_EMBEDDING_FUNCTION: Optional[embedding_functions.EmbeddingFunction] = None
_CONTEXT_RULES_ADDED = False
_NEGATION_RULES_ADDED = False

PRIMARY_RERANKER_NAME = os.getenv(
    "ONCORAGGRAPH_RERANKER_PRIMARY_MODEL",
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
)
SECONDARY_RERANKER_NAME = os.getenv(
    "ONCORAGGRAPH_RERANKER_SECONDARY_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

_EMBEDDER_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

_PIPELINE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "system_config.yaml"
if _PIPELINE_CONFIG_PATH.exists():
    try:
        _PIPELINE_CONFIG = yaml.safe_load(_PIPELINE_CONFIG_PATH.read_text()) or {}
    except Exception:
        _PIPELINE_CONFIG = {}
else:
    _PIPELINE_CONFIG = {}

_RUNTIME_DEFAULTS = _PIPELINE_CONFIG.get("runtime_defaults", {}) if isinstance(_PIPELINE_CONFIG, dict) else {}
_PRIMARY_OVERRIDE = _RUNTIME_DEFAULTS.get("reranker_primary_model")
_SECONDARY_OVERRIDE = _RUNTIME_DEFAULTS.get("reranker_secondary_model")
_SECONDARY_ENABLED = _RUNTIME_DEFAULTS.get("reranker_secondary_enabled", True)
_RERANKER_BATCH_SIZE = _RUNTIME_DEFAULTS.get("reranker_batch_size")
_HF_LOCAL_ONLY = _RUNTIME_DEFAULTS.get("huggingface_local_files_only")
_HF_CACHE_FOLDER = _RUNTIME_DEFAULTS.get("huggingface_cache_folder")

if _PRIMARY_OVERRIDE:
    PRIMARY_RERANKER_NAME = _PRIMARY_OVERRIDE
if _SECONDARY_OVERRIDE:
    SECONDARY_RERANKER_NAME = _SECONDARY_OVERRIDE
if not _SECONDARY_ENABLED:
    SECONDARY_RERANKER_NAME = ""


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _hf_cache_folder() -> Optional[str]:
    if _HF_CACHE_FOLDER:
        return str(_HF_CACHE_FOLDER)
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return str(Path(hf_home).expanduser().resolve() / "hub")
    return str((Path.home() / ".cache" / "huggingface" / "hub").resolve())


def _model_cache_exists(model_name: str) -> bool:
    cache_folder = _hf_cache_folder()
    if not cache_folder:
        return False
    cache_root = Path(cache_folder)
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    return repo_dir.exists()


def _local_files_only(model_name: str) -> bool:
    if _HF_LOCAL_ONLY is not None:
        return _as_bool(_HF_LOCAL_ONLY, default=False)
    return _model_cache_exists(model_name)

_NEGATION_SCOPE = _PIPELINE_CONFIG.get("negation_detection", {}).get("scope_tokens", 5)
_NEGATION_TRIGGERS = _PIPELINE_CONFIG.get("negation_detection", {}).get("triggers", [])

_CUSTOM_CONTEXT_RULES = [
    ConTextRule(
        "plan for",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "planned for",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "planned",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=3,
    ),
    ConTextRule(
        "planning for",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "plan is for",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "will undergo",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=8,
    ),
    ConTextRule(
        "will receive",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=8,
    ),
    ConTextRule(
        "scheduled for",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=8,
    ),
    ConTextRule(
        "scheduled to",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=8,
    ),
    ConTextRule(
        "candidate for",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "consider",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "considered for",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "recommend",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "recommended for",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "need for",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "needs to undergo",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=8,
    ),
    ConTextRule(
        "to undergo",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "discussed potential",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=8,
    ),
    ConTextRule(
        "discussing potential",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=8,
    ),
    ConTextRule(
        "discussing anticipated",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=8,
    ),
    ConTextRule(
        "anticipated side effects",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "potential side effects",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=6,
    ),
    ConTextRule(
        "education on side effects",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=8,
    ),
    ConTextRule(
        "anticipatory guidance",
        "HYPOTHETICAL",
        direction="FORWARD",
        max_scope=8,
    ),
]

_NEGATION_CONTEXT_RULES = [
    ConTextRule(trigger, "NEGATED_EXISTENCE", direction="FORWARD", max_scope=_NEGATION_SCOPE)
    for trigger in _NEGATION_TRIGGERS
]


def _select_device(env_var: str, default_index: int) -> str:
    """Decide which device to target for a model."""
    if os.getenv("ONCORAGGRAPH_FORCE_CPU"):
        return "cpu"
    requested = os.getenv(env_var)
    if requested:
        requested = requested.strip().lower()
        if requested == "cpu":
            return "cpu"
        if requested.startswith("cuda"):
            return requested
        if requested.isdigit():
            return f"cuda:{requested}"

    if torch is None or not torch.cuda.is_available():
        return "cpu"

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return "cpu"

    if default_index < gpu_count:
        return f"cuda:{default_index}"
    return "cuda:0"


def _configure_torch_threads() -> None:
    threads = os.getenv("ONCORAGGRAPH_CPU_THREADS")
    if threads:
        try:
            value = max(1, int(threads))
            if torch is not None:
                torch.set_num_threads(value)
            os.environ["OMP_NUM_THREADS"] = str(value)
            log(f"CPU thread pool set to {value}", level="INFO", debug=True)
        except ValueError:
            log(f"Ignoring invalid ONCORAGGRAPH_CPU_THREADS={threads}", level="WARNING")


def _ensure_custom_context_rules(nlp) -> None:
    """Inject custom medspaCy context rules for temporal and negation control."""
    global _CONTEXT_RULES_ADDED, _NEGATION_RULES_ADDED

    try:
        context_component = nlp.get_pipe("medspacy_context")
    except (KeyError, AttributeError):
        log(
            "medspaCy context component not available; skipping custom rules injection.",
            level="WARNING",
        )
        return

    if not _CONTEXT_RULES_ADDED and _CUSTOM_CONTEXT_RULES:
        try:
            context_component.add(_CUSTOM_CONTEXT_RULES)
            _CONTEXT_RULES_ADDED = True
            log(
                f"Added {len(_CUSTOM_CONTEXT_RULES)} custom HYPOTHETICAL context rules.",
                level="INFO",
                debug=True,
            )
        except Exception as exc:
            log(
                f"Failed to add custom medspaCy context rules: {exc}",
                level="WARNING",
            )

    if not _NEGATION_RULES_ADDED and _NEGATION_CONTEXT_RULES:
        try:
            context_component.add(_NEGATION_CONTEXT_RULES)
            _NEGATION_RULES_ADDED = True
            log(
                f"Added {len(_NEGATION_CONTEXT_RULES)} custom NEGATION context rules.",
                level="INFO",
                debug=True,
            )
        except Exception as exc:
            log(
                f"Failed to add negation context rules: {exc}",
                level="WARNING",
            )


class SentenceTransformerEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Custom embedding function wrapping a SentenceTransformer instance."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.batch_size = max(
            8, int(os.getenv("ONCORAGGRAPH_ST_BATCH_SIZE", "32") or 32)
        )

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()


def initialize_reranker(
    device_override: Optional[str] = None,
    force_reload: bool = False,
    prefer_lightweight: bool = False,
) -> CrossEncoder:
    """Ensure both cross-encoder rerankers are loaded, optionally reinitializing."""
    del prefer_lightweight  # Parameter retained for backwards compatibility.
    global RERANKER_PRIMARY, RERANKER_SECONDARY

    if force_reload:
        if RERANKER_PRIMARY is not None:
            try:
                RERANKER_PRIMARY.model = None  # type: ignore[attr-defined]
            except AttributeError:
                pass
        if RERANKER_SECONDARY is not None and RERANKER_SECONDARY is not RERANKER_PRIMARY:
            try:
                RERANKER_SECONDARY.model = None  # type: ignore[attr-defined]
            except AttributeError:
                pass
        RERANKER_PRIMARY = None
        RERANKER_SECONDARY = None

    device = device_override or _select_device("ONCORAGGRAPH_RERANKER_DEVICE", 0)

    def _load_cross_encoder(model_name: str, target_device: str) -> CrossEncoder:
        local_only = _local_files_only(model_name)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Some weights.*were not initialized.*")
            warnings.filterwarnings("ignore", message=".*You should probably TRAIN.*")
            return CrossEncoder(
                model_name,
                device=target_device,
                cache_folder=_hf_cache_folder(),
                local_files_only=local_only,
            )

    if RERANKER_PRIMARY is None:
        log(f"Loading primary cross-encoder reranker on {device}...", level="STEP")
        try:
            RERANKER_PRIMARY = _load_cross_encoder(PRIMARY_RERANKER_NAME, device)
        except Exception as exc:
            msg = str(exc).lower()
            cuda_like = any(token in msg for token in ["nvidia driver", "cuda", "cublas", "hip", "driver on your system is too old"])
            if device != "cpu" and cuda_like:
                log(
                    f"Primary reranker failed on {device}: {exc}. Falling back to CPU.",
                    level="WARNING",
                )
                if torch is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                RERANKER_PRIMARY = _load_cross_encoder(PRIMARY_RERANKER_NAME, "cpu")
            else:
                raise
        log(f"Primary reranker loaded ({PRIMARY_RERANKER_NAME})", level="SUCCESS")

    if not SECONDARY_RERANKER_NAME:
        RERANKER_SECONDARY = None
    elif RERANKER_SECONDARY is None:
        secondary_device = device_override or device
        if SECONDARY_RERANKER_NAME == PRIMARY_RERANKER_NAME:
            RERANKER_SECONDARY = RERANKER_PRIMARY
        else:
            try:
                log(
                    f"Loading secondary cross-encoder reranker on {secondary_device}...",
                    level="STEP",
                )
                try:
                    RERANKER_SECONDARY = _load_cross_encoder(SECONDARY_RERANKER_NAME, secondary_device)
                except Exception as exc:
                    msg = str(exc).lower()
                    cuda_like = any(token in msg for token in ["nvidia driver", "cuda", "cublas", "hip", "driver on your system is too old"])
                    if secondary_device != "cpu" and cuda_like:
                        log(
                            f"Secondary reranker failed on {secondary_device}: {exc}. Falling back to CPU.",
                            level="WARNING",
                        )
                        if torch is not None and torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        RERANKER_SECONDARY = _load_cross_encoder(SECONDARY_RERANKER_NAME, "cpu")
                    else:
                        raise
                log(
                    f"Secondary reranker loaded ({SECONDARY_RERANKER_NAME})",
                    level="SUCCESS",
                )
            except Exception as exc:
                log(
                    f"Failed to load secondary reranker '{SECONDARY_RERANKER_NAME}': {exc}",
                    level="WARNING",
                )
                RERANKER_SECONDARY = None

    return RERANKER_PRIMARY


def _predict_scores(model: Optional[CrossEncoder], pairs: list[list[str]]) -> Optional[list[float]]:
    if model is None:
        return None
    try:
        batch_size = None
        try:
            if _RERANKER_BATCH_SIZE is not None and str(_RERANKER_BATCH_SIZE).strip() != "":
                batch_size = int(_RERANKER_BATCH_SIZE)
        except Exception:
            batch_size = None
        if batch_size:
            raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        else:
            raw_scores = model.predict(pairs)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        return [float(score) for score in raw_scores]
    except Exception as exc:
        log(f"Reranker prediction failed: {exc}", level="WARNING")
        return None


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    mn = min(scores)
    mx = max(scores)
    if mx - mn <= 1e-8:
        return [0.0 for _ in scores]
    return [(val - mn) / (mx - mn) for val in scores]


def get_combined_reranker_scores(
    pairs: list[list[str]],
    device_override: Optional[str] = None,
    force_reload: bool = False,
    prefer_lightweight: bool = False,
) -> list[float]:
    """Return blended semantic scores from both rerankers."""
    initialize_reranker(device_override=device_override, force_reload=force_reload, prefer_lightweight=prefer_lightweight)
    primary_scores = _predict_scores(RERANKER_PRIMARY, pairs)
    if not SECONDARY_RERANKER_NAME:
        if primary_scores is None:
            raise RuntimeError("Unable to obtain reranker scores from primary model")
        return primary_scores
    secondary_scores = _predict_scores(RERANKER_SECONDARY, pairs)

    if primary_scores is None and secondary_scores is None:
        raise RuntimeError("Unable to obtain reranker scores from any model")

    if primary_scores is None:
        return [float(s) for s in secondary_scores]  # type: ignore[arg-type]
    if secondary_scores is None or len(secondary_scores) != len(primary_scores):
        return primary_scores

    norm_primary = _normalize_scores(primary_scores)
    norm_secondary = _normalize_scores([float(s) for s in secondary_scores])
    blended = [
        (norm_primary[i] + norm_secondary[i]) / 2.0
        for i in range(len(norm_primary))
    ]
    return blended


def initialize_models() -> None:
    """Load long-lived NLP models used by the extraction pipeline."""
    global NLP_MED, CLINICAL_EMBEDDER, CHROMA_EMBEDDER, CHROMA_EMBEDDING_FUNCTION

    if (
        NLP_MED is not None
        and CLINICAL_EMBEDDER is not None
        and CHROMA_EMBEDDER is not None
        and CHROMA_EMBEDDING_FUNCTION is not None
    ):
        _ensure_custom_context_rules(NLP_MED)
        return

    _configure_torch_threads()

    log("INITIALIZING MODELS", level="HEADER")

    initialize_reranker()

    if NLP_MED is None:
        log("Loading medspaCy for clinical context detection...", level="STEP")
        NLP_MED = medspacy.load()
        log("medspaCy loaded", level="SUCCESS")
        _ensure_custom_context_rules(NLP_MED)
    else:
        _ensure_custom_context_rules(NLP_MED)

    if CLINICAL_EMBEDDER is None:
        clinical_device = _select_device("ONCORAGGRAPH_CLINICAL_EMBEDDER_DEVICE", 1)
        log(f"Loading SapBERT clinical embedder on {clinical_device}...", level="STEP")
        # Always instantiate on CPU first to guarantee a usable copy if GPU placement fails.
        # Suppress sentence-transformers info messages about model creation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*No sentence-transformers model found.*")
            CLINICAL_EMBEDDER = SentenceTransformer(
                _EMBEDDER_MODEL_NAME,
                device="cpu",
                cache_folder=_hf_cache_folder(),
                local_files_only=_local_files_only(_EMBEDDER_MODEL_NAME),
            )
        target_device = "cpu"
        if clinical_device != "cpu":
            try:
                CLINICAL_EMBEDDER.to(clinical_device)
                target_device = clinical_device
            except Exception as exc:
                log(
                    f"Failed to load SapBERT clinical embedder on {clinical_device}: {exc}. Falling back to CPU.",
                    level="WARNING",
                )
                if torch is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                # Re-initialize on CPU to avoid partial CUDA state lingering.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*No sentence-transformers model found.*")
                    CLINICAL_EMBEDDER = SentenceTransformer(
                        _EMBEDDER_MODEL_NAME,
                        device="cpu",
                        cache_folder=_hf_cache_folder(),
                        local_files_only=_local_files_only(_EMBEDDER_MODEL_NAME),
                    )
                target_device = "cpu"
        log(
            f"SapBERT embedder loaded for entity deduplication (device={target_device})",
            level="SUCCESS",
        )

    if CHROMA_EMBEDDER is None:
        chroma_device = _select_device("ONCORAGGRAPH_CHROMA_EMBEDDER_DEVICE", 2)
        log(f"Loading SapBERT Chroma embedder on {chroma_device}...", level="STEP")
        # Suppress sentence-transformers info messages about model creation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*No sentence-transformers model found.*")
            CHROMA_EMBEDDER = SentenceTransformer(
                _EMBEDDER_MODEL_NAME,
                device="cpu",
                cache_folder=_hf_cache_folder(),
                local_files_only=_local_files_only(_EMBEDDER_MODEL_NAME),
            )
        chroma_target_device = "cpu"
        if chroma_device != "cpu":
            try:
                CHROMA_EMBEDDER.to(chroma_device)
                chroma_target_device = chroma_device
            except Exception as exc:
                log(
                    f"Failed to load SapBERT Chroma embedder on {chroma_device}: {exc}. Falling back to CPU.",
                    level="WARNING",
                )
                if torch is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*No sentence-transformers model found.*")
                    CHROMA_EMBEDDER = SentenceTransformer(
                        _EMBEDDER_MODEL_NAME,
                        device="cpu",
                        cache_folder=_hf_cache_folder(),
                        local_files_only=_local_files_only(_EMBEDDER_MODEL_NAME),
                    )
                chroma_target_device = "cpu"
        CHROMA_EMBEDDING_FUNCTION = SentenceTransformerEmbeddingFunction(CHROMA_EMBEDDER)
        log(
            f"SapBERT embedder loaded for ChromaDB indexing (device={chroma_target_device})",
            level="SUCCESS",
        )

    log("All models initialized", level="SUCCESS")


def get_scispacy_model(model_name: str):
    """Load and cache scispaCy models by name."""
    if model_name not in MODEL_CACHE:
        log(f"Loading scispaCy model: {model_name}...", level="STEP", debug=True)
        try:
            MODEL_CACHE[model_name] = spacy.load(model_name)
            log(f"Model {model_name} loaded and cached", level="SUCCESS", debug=True)
        except OSError:
            log(f"Model {model_name} not found. Installing...", level="WARNING")
            subprocess.run(
                [
                    "pip",
                    "install",
                    f"https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{model_name}-0.5.4.tar.gz",
                ],
                check=False,
            )
            MODEL_CACHE[model_name] = spacy.load(model_name)
            log(f"Model {model_name} installed and loaded", level="SUCCESS")

    return MODEL_CACHE[model_name]


def get_chroma_embedding_function() -> embedding_functions.EmbeddingFunction:
    """Return the embedding function configured for ChromaDB."""
    global CHROMA_EMBEDDING_FUNCTION
    if CHROMA_EMBEDDING_FUNCTION is None:
        initialize_models()
    return CHROMA_EMBEDDING_FUNCTION


__all__ = [
    "RERANKER_PRIMARY",
    "RERANKER_SECONDARY",
    "MODEL_CACHE",
    "NLP_MED",
    "CLINICAL_EMBEDDER",
    "initialize_reranker",
    "get_combined_reranker_scores",
    "initialize_models",
    "get_scispacy_model",
    "get_chroma_embedding_function",
]
def move_clinical_embedder_to_cpu(force_reload: bool = False) -> SentenceTransformer:
    """Ensure the clinical embedder is resident on CPU (reloading if required)."""
    global CLINICAL_EMBEDDER

    log("Falling back to CPU for SapBERT clinical embedder...", level="WARNING")
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    if CLINICAL_EMBEDDER is None or force_reload:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*No sentence-transformers model found.*")
            CLINICAL_EMBEDDER = SentenceTransformer(
                _EMBEDDER_MODEL_NAME,
                device="cpu",
                cache_folder=_hf_cache_folder(),
                local_files_only=_local_files_only(_EMBEDDER_MODEL_NAME),
            )
        return CLINICAL_EMBEDDER

    try:
        CLINICAL_EMBEDDER.to("cpu")
    except Exception:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*No sentence-transformers model found.*")
            CLINICAL_EMBEDDER = SentenceTransformer(
                _EMBEDDER_MODEL_NAME,
                device="cpu",
                cache_folder=_hf_cache_folder(),
                local_files_only=_local_files_only(_EMBEDDER_MODEL_NAME),
            )
        return CLINICAL_EMBEDDER
