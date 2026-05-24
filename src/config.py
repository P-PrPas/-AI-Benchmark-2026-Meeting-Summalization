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


def _resolve_llm_model_path(model_dir: Path) -> Path:
    """Pick a real local LLM directory and never fall back to Hub implicitly."""
    env_value = os.environ.get("CAMNET_LLM_MODEL_PATH")
    if env_value:
        candidate = Path(env_value).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate

    model_name = os.environ.get("CAMNET_LLM_MODEL_NAME")
    if model_name:
        return model_dir / model_name

    preferred = model_dir / "Qwen2.5-7B-Instruct"
    if preferred.exists():
        return preferred

    candidates = sorted(
        path for path in model_dir.iterdir()
        if path.is_dir() and path.name != "bge-m3"
    )
    if len(candidates) == 1:
        return candidates[0]

    return preferred


BASE_DIR = PROJECT_ROOT
DATA_DIR = _resolve_path("CAMNET_DATA_DIR", PROJECT_ROOT / "data")
MODEL_DIR = _resolve_path(
    "CAMNET_MODEL_DIR",
    _prefer_existing_path(Path("/model/weights"), PROJECT_ROOT / "weight"),
)
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
BGE_MODEL_PATH = _resolve_path("CAMNET_BGE_MODEL_PATH", MODEL_DIR / "bge-m3")
LLM_MODEL_PATH = _resolve_llm_model_path(MODEL_DIR)

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

RETRIEVAL_TOP_K = 10
REFERENCE_TOP_N = 3
