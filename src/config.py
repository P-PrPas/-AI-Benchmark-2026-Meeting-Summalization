import os

BASE_DIR = "/project/zz991000-zdeva/zz991011/CAMNET_P"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = "/project/zz991000-zdeva/zz991011/models"

TRAIN_PATH = os.path.join(DATA_DIR, "train_set.json")
TEST_PATH = os.path.join(DATA_DIR, "test_set.json")
BGE_MODEL_PATH = os.path.join(MODEL_DIR, "bge-m3")
LLM_MODEL_PATH = os.path.join(MODEL_DIR, "Qwen3.6-27B-unsloth")

EVAL_SAMPLE_DIR = os.path.join(DATA_DIR, "eval_sample")
SUBMISSION_PATH = os.path.join(EVAL_SAMPLE_DIR, "submission.csv")

RETRIEVAL_TOP_K = 10
REFERENCE_TOP_N = 3

OUTPUT_DIR = "./output"