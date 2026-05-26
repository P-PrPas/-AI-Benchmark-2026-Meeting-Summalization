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
        candidate = Path("/project/zz991000-zdeva/zz991011/models")
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
EMBED_MODEL_PATH = _resolve_embed_model_path("Qwen3-Embedding-8B")
LLM_MODEL_PATH = _resolve_model_path("typhoon2.5-qwen3-4b")

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

RETRIEVAL_TOP_K = int(os.environ.get("CAMNET_RETRIEVAL_TOP_K", "12"))
RETRIEVAL_CANDIDATE_K = int(os.environ.get("CAMNET_RETRIEVAL_CANDIDATE_K", str(RETRIEVAL_TOP_K)))
REFERENCE_TOP_N = int(os.environ.get("CAMNET_REFERENCE_TOP_N", "3"))
GENERATOR_CONTEXT_K_FACT = int(os.environ.get("CAMNET_GENERATOR_CONTEXT_K_FACT", "4"))
GENERATOR_CONTEXT_K_AGGREGATE = int(os.environ.get("CAMNET_GENERATOR_CONTEXT_K_AGGREGATE", "6"))

GENERATOR_MAX_SEQ_LEN = int(os.environ.get("CAMNET_GENERATOR_MAX_SEQ_LEN", "8192"))
FACT_MAX_NEW_TOKENS = int(os.environ.get("CAMNET_FACT_MAX_NEW_TOKENS", "192"))
AGGREGATE_MAX_NEW_TOKENS = int(os.environ.get("CAMNET_AGGREGATE_MAX_NEW_TOKENS", "320"))
STRICT_RETRY_MAX_NEW_TOKENS = int(os.environ.get("CAMNET_STRICT_RETRY_MAX_NEW_TOKENS", "96"))
FACT_MAX_ANSWER_CHARS = int(os.environ.get("CAMNET_FACT_MAX_ANSWER_CHARS", "420"))
DEFAULT_REPETITION_PENALTY = float(os.environ.get("CAMNET_REPETITION_PENALTY", "1.05"))
STRICT_REPETITION_PENALTY = float(os.environ.get("CAMNET_STRICT_REPETITION_PENALTY", "1.08"))

HYBRID_DENSE_WEIGHT = float(os.environ.get("CAMNET_HYBRID_DENSE_WEIGHT", "0.8"))
HYBRID_LEXICAL_WEIGHT = float(os.environ.get("CAMNET_HYBRID_LEXICAL_WEIGHT", "0.2"))
