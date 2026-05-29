import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(env_name: str, default_path: Path) -> Path:
    """Resolve a path from env or fallback to a project-relative default."""
    raw_value = os.environ.get(env_name)
    path = Path(raw_value).expanduser() if raw_value else default_path
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _prefer_existing_path(*paths: Path) -> Path:
    """Return the first existing path, otherwise fall back to the last candidate."""
    for path in paths:
        if path.exists():
            return path
    return paths[-1]


def _discover_test_path() -> Path:
    """Resolve the runtime test file without assuming a fixed server filename."""
    env_value = os.environ.get("CAMNET_TEST_PATH")
    if env_value:
        candidate = Path(env_value).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate

    candidates = [
        Path("/model/test/test_set.json"),
        PROJECT_ROOT / "data" / "test" / "test_set.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for test_dir in (Path("/model/test"), PROJECT_ROOT / "data" / "test"):
        if test_dir.exists():
            json_files = sorted(test_dir.glob("*.json"))
            if json_files:
                named_candidates = [
                    path for path in json_files
                    if "test" in path.name.lower() or "query" in path.name.lower()
                ]
                if named_candidates:
                    return max(named_candidates, key=lambda path: path.stat().st_size)
                return max(json_files, key=lambda path: path.stat().st_size)

    return Path("/model/test/test_set.json")


def _resolve_model_dir() -> Path:
    """Resolve the shared model root used by finetune and inference."""
    env_value = os.environ.get("CAMNET_MODEL_DIR")
    if env_value:
        candidate = Path(env_value).expanduser()
    else:
        candidate = _prefer_existing_path(Path("/model/weights"), PROJECT_ROOT / "weight")
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return candidate


def _resolve_model_path(default_name: str) -> Path:
    """Resolve a model path without probing the filesystem at import time."""
    env_value = os.environ.get("CAMNET_LLM_MODEL_PATH")
    if env_value:
        candidate = Path(env_value).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate

    model_name = os.environ.get("CAMNET_LLM_MODEL_NAME")
    model_dir = MODEL_DIR
    if model_name:
        return model_dir / model_name
    return model_dir / default_name


def _resolve_embed_model_path(default_name: str) -> Path:
    """Resolve the embedding model path."""
    env_value = os.environ.get("CAMNET_EMBED_MODEL_PATH")
    if env_value:
        candidate = Path(env_value).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate
    model_name = os.environ.get("CAMNET_EMBED_MODEL_NAME")
    if model_name:
        return MODEL_DIR / model_name
    return MODEL_DIR / default_name


def _resolve_rerank_model_path(default_name: str) -> Path:
    """Resolve the reranker model path."""
    env_value = os.environ.get("CAMNET_RERANK_MODEL_PATH")
    if env_value:
        candidate = Path(env_value).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate
    model_name = os.environ.get("CAMNET_RERANK_MODEL_NAME")
    if model_name:
        return MODEL_DIR / model_name
    return MODEL_DIR / default_name


BASE_DIR = PROJECT_ROOT
DATA_DIR = _resolve_path("CAMNET_DATA_DIR", PROJECT_ROOT / "data")
MODEL_DIR = _resolve_model_dir()
OUTPUT_DIR = _resolve_path(
    "CAMNET_OUTPUT_DIR",
    _prefer_existing_path(Path("/result"), PROJECT_ROOT / "output"),
)

TRAIN_PATH = _resolve_path(
    "CAMNET_TRAIN_PATH",
    DATA_DIR / "train" / "train_set.json",
)
TEST_PATH = _resolve_path(
    "CAMNET_TEST_PATH",
    _discover_test_path(),
)
EMBED_MODEL_PATH = _resolve_embed_model_path("bge-m3")
RERANK_MODEL_PATH = _resolve_rerank_model_path("reranker_phase_b_v1_final_model")
LLM_MODEL_PATH = _resolve_model_path("llm_best_run_c5_final_merged")

EVAL_SAMPLE_DIR = _resolve_path("CAMNET_EVAL_SAMPLE_DIR", DATA_DIR / "eval_sample")
SUBMISSION_PATH = _resolve_path(
    "CAMNET_SUBMISSION_PATH",
    EVAL_SAMPLE_DIR / "submission.csv",
)

PROGRESS_LIB = _resolve_path(
    "CAMNET_PROGRESS_LIB",
    _prefer_existing_path(
        Path("/benchmark_lib/progress"),
        PROJECT_ROOT / "benchmark_lib" / "progress",
    ),
)

STARTUP_SLEEP_SECONDS = int(os.environ.get("CAMNET_STARTUP_SLEEP_SECONDS", "10"))
EMBED_BATCH_SIZE = int(os.environ.get("CAMNET_EMBED_BATCH_SIZE", "128"))
RERANK_BATCH_SIZE = int(os.environ.get("CAMNET_RERANK_BATCH_SIZE", "16"))
RERANK_MAX_LENGTH = int(os.environ.get("CAMNET_RERANK_MAX_LENGTH", "2048"))
GENERATOR_BATCH_SIZE = int(os.environ.get("CAMNET_GENERATOR_BATCH_SIZE", "4"))
ENABLE_FACT_FEW_SHOT = os.environ.get("CAMNET_ENABLE_FACT_FEW_SHOT", "1").strip() not in {
    "0",
    "false",
    "False",
}
ENABLE_LIST_FEW_SHOT = os.environ.get("CAMNET_ENABLE_LIST_FEW_SHOT", "1").strip() not in {
    "0",
    "false",
    "False",
}
ENABLE_SYNTHESIS_FEW_SHOT = os.environ.get("CAMNET_ENABLE_SYNTHESIS_FEW_SHOT", "1").strip() not in {
    "0",
    "false",
    "False",
}
ENABLE_LLM_REF_ARBITER = os.environ.get("CAMNET_ENABLE_LLM_REF_ARBITER", "0").strip() not in {
    "0",
    "false",
    "False",
}
REF_ARBITER_MAX_CANDIDATES = int(os.environ.get("CAMNET_REF_ARBITER_MAX_CANDIDATES", "3"))
REF_ARBITER_TRIGGER_MODE = os.environ.get("CAMNET_REF_ARBITER_TRIGGER_MODE", "ambiguous_only").strip().lower()
REF_ARBITER_MAX_NEW_TOKENS = int(os.environ.get("CAMNET_REF_ARBITER_MAX_NEW_TOKENS", "48"))
ENABLE_FACT_ANSWER_REWRITE = os.environ.get("CAMNET_ENABLE_FACT_ANSWER_REWRITE", "0").strip() not in {
    "0",
    "false",
    "False",
}
FACT_REWRITE_MAX_NEW_TOKENS = int(os.environ.get("CAMNET_FACT_REWRITE_MAX_NEW_TOKENS", "96"))
FACT_REWRITE_TRIGGER_CHARS = int(os.environ.get("CAMNET_FACT_REWRITE_TRIGGER_CHARS", "180"))
FACT_REWRITE_MIN_SOURCE_OVERLAP = float(os.environ.get("CAMNET_FACT_REWRITE_MIN_SOURCE_OVERLAP", "0.50"))
ENABLE_LEARNED_REF_SELECTOR = os.environ.get("CAMNET_ENABLE_LEARNED_REF_SELECTOR", "0").strip() not in {
    "0",
    "false",
    "False",
}
REF_SELECTOR_MODEL_PATH = os.environ.get("CAMNET_REF_SELECTOR_MODEL_PATH")
ENABLE_SOURCE_ANCHORED_FACT_TARGETS = os.environ.get("CAMNET_ENABLE_SOURCE_ANCHORED_FACT_TARGETS", "0").strip() not in {
    "0",
    "false",
    "False",
}
SOURCE_ANCHORED_FACT_MIN_OVERLAP = float(os.environ.get("CAMNET_SOURCE_ANCHORED_FACT_MIN_OVERLAP", "0.45"))

RETRIEVAL_TOP_K = int(os.environ.get("CAMNET_RETRIEVAL_TOP_K", "12"))
RETRIEVAL_CANDIDATE_K = int(os.environ.get("CAMNET_RETRIEVAL_CANDIDATE_K", str(RETRIEVAL_TOP_K)))
RERANK_TOP_K = int(os.environ.get("CAMNET_RERANK_TOP_K", "20"))
REFERENCE_TOP_N = int(os.environ.get("CAMNET_REFERENCE_TOP_N", "3"))
REFERENCE_TOP_N_MAX = int(os.environ.get("CAMNET_REFERENCE_TOP_N_MAX", "4"))
USE_RERANKER = os.environ.get("CAMNET_USE_RERANKER", "0").strip() not in {"0", "false", "False"}
ENABLE_ADAPTIVE_RERANKING = os.environ.get("CAMNET_ENABLE_ADAPTIVE_RERANKING", "1").strip() not in {
    "0",
    "false",
    "False",
}
ENABLE_DYNAMIC_REF_SELECTION = os.environ.get("CAMNET_ENABLE_DYNAMIC_REF_SELECTION", "0").strip() not in {
    "0",
    "false",
    "False",
}
ENABLE_QUERY_REFINEMENT = os.environ.get("CAMNET_ENABLE_QUERY_REFINEMENT", "0").strip() not in {
    "0",
    "false",
    "False",
}
ENABLE_EVIDENCE_COMPRESSION = os.environ.get("CAMNET_ENABLE_EVIDENCE_COMPRESSION", "0").strip() not in {
    "0",
    "false",
    "False",
}
RERANK_INSTRUCTION = os.environ.get(
    "CAMNET_RERANK_INSTRUCTION",
    "Given a web search query, retrieve relevant passages that answer the query",
)
GENERATOR_CONTEXT_K_FACT = int(os.environ.get("CAMNET_GENERATOR_CONTEXT_K_FACT", "3"))
GENERATOR_CONTEXT_K_AGGREGATE = int(os.environ.get("CAMNET_GENERATOR_CONTEXT_K_AGGREGATE", "5"))
GENERATOR_CONTEXT_K_SYNTHESIS = int(os.environ.get("CAMNET_GENERATOR_CONTEXT_K_SYNTHESIS", "6"))

GENERATOR_MAX_SEQ_LEN = int(os.environ.get("CAMNET_GENERATOR_MAX_SEQ_LEN", "8192"))
FACT_MAX_NEW_TOKENS = int(os.environ.get("CAMNET_FACT_MAX_NEW_TOKENS", "160"))
AGGREGATE_MAX_NEW_TOKENS = int(os.environ.get("CAMNET_AGGREGATE_MAX_NEW_TOKENS", "256"))
SYNTHESIS_MAX_NEW_TOKENS = int(os.environ.get("CAMNET_SYNTHESIS_MAX_NEW_TOKENS", "288"))
STRICT_RETRY_MAX_NEW_TOKENS = int(os.environ.get("CAMNET_STRICT_RETRY_MAX_NEW_TOKENS", "96"))
FACT_MAX_ANSWER_CHARS = int(os.environ.get("CAMNET_FACT_MAX_ANSWER_CHARS", "320"))
DEFAULT_REPETITION_PENALTY = float(os.environ.get("CAMNET_REPETITION_PENALTY", "1.04"))
STRICT_REPETITION_PENALTY = float(os.environ.get("CAMNET_STRICT_REPETITION_PENALTY", "1.08"))

HYBRID_DENSE_WEIGHT = float(os.environ.get("CAMNET_HYBRID_DENSE_WEIGHT", "0.8"))
HYBRID_LEXICAL_WEIGHT = float(os.environ.get("CAMNET_HYBRID_LEXICAL_WEIGHT", "0.2"))
REF_SELECTION_TOP2_MIN = float(os.environ.get("CAMNET_REF_SELECTION_TOP2_MIN", "0.22"))
REF_SELECTION_TOP3_MIN = float(os.environ.get("CAMNET_REF_SELECTION_TOP3_MIN", "0.16"))
REF_SELECTION_FACT_MAX_GAP = float(os.environ.get("CAMNET_REF_SELECTION_FACT_MAX_GAP", "0.18"))
REF_SELECTION_AGG_MAX_GAP = float(os.environ.get("CAMNET_REF_SELECTION_AGG_MAX_GAP", "0.24"))
REF_SELECTION_LOW_CONFIDENCE = float(os.environ.get("CAMNET_REF_SELECTION_LOW_CONFIDENCE", "0.34"))
REF_SELECTION_FACT_TOP2_MIN = float(os.environ.get("CAMNET_REF_SELECTION_FACT_TOP2_MIN", str(REF_SELECTION_TOP2_MIN)))
REF_SELECTION_LIST_TOP2_MIN = float(os.environ.get("CAMNET_REF_SELECTION_LIST_TOP2_MIN", str(REF_SELECTION_TOP2_MIN)))
REF_SELECTION_LIST_TOP3_MIN = float(os.environ.get("CAMNET_REF_SELECTION_LIST_TOP3_MIN", str(REF_SELECTION_TOP3_MIN)))
REF_SELECTION_SYNTH_TOP2_MIN = float(os.environ.get("CAMNET_REF_SELECTION_SYNTH_TOP2_MIN", str(REF_SELECTION_TOP2_MIN)))
REF_SELECTION_SYNTH_TOP3_MIN = float(os.environ.get("CAMNET_REF_SELECTION_SYNTH_TOP3_MIN", str(REF_SELECTION_TOP3_MIN)))
REF_SELECTION_LIST_MAX_GAP = float(os.environ.get("CAMNET_REF_SELECTION_LIST_MAX_GAP", str(REF_SELECTION_AGG_MAX_GAP)))
REF_SELECTION_SYNTH_MAX_GAP = float(os.environ.get("CAMNET_REF_SELECTION_SYNTH_MAX_GAP", str(REF_SELECTION_AGG_MAX_GAP)))
QUERY_REFINEMENT_MAX_ENTROPY = float(os.environ.get("CAMNET_QUERY_REFINEMENT_MAX_ENTROPY", "0.92"))
ADAPTIVE_RERANK_FACT_TOP1_MIN = float(os.environ.get("CAMNET_ADAPTIVE_RERANK_FACT_TOP1_MIN", "0.72"))
ADAPTIVE_RERANK_FACT_MIN_GAP = float(os.environ.get("CAMNET_ADAPTIVE_RERANK_FACT_MIN_GAP", "0.18"))
ADAPTIVE_RERANK_FACT_MAX_ENTROPY = float(os.environ.get("CAMNET_ADAPTIVE_RERANK_FACT_MAX_ENTROPY", "0.78"))
PROGRESS_UPDATE_EVERY = int(os.environ.get("CAMNET_PROGRESS_UPDATE_EVERY", "10"))
