"""Central configuration. All values are overridable via environment variables."""
import os

SEED = int(os.environ.get("GRAPHMIND_SEED", 42))

# ---- Graph construction ----
N_TRIPLES = int(os.environ.get("GRAPHMIND_N_TRIPLES", 220))

# ---- Question generation ----
N_EVAL_QUESTIONS = int(os.environ.get("GRAPHMIND_N_QUESTIONS", 150))
MIN_HOPS = int(os.environ.get("GRAPHMIND_MIN_HOPS", 2))
MAX_HOPS = int(os.environ.get("GRAPHMIND_MAX_HOPS", 3))

# ---- Train/val/test split ----
N_TEST = int(os.environ.get("GRAPHMIND_N_TEST", 30))
N_VAL = int(os.environ.get("GRAPHMIND_N_VAL", 20))
SPLIT_SEED = int(os.environ.get("GRAPHMIND_SPLIT_SEED", 2024))

# ---- Feature engineering ----
RANDOM_FEAT_DIM = int(os.environ.get("GRAPHMIND_FEAT_DIM", 24))
RELATION_NOISE_P = float(os.environ.get("GRAPHMIND_NOISE_P", 0.3))

# ---- Model / training ----
HIDDEN_DIM = int(os.environ.get("GRAPHMIND_HIDDEN_DIM", 32))
GAT_HEADS = int(os.environ.get("GRAPHMIND_GAT_HEADS", 4))
N_LAYERS = int(os.environ.get("GRAPHMIND_N_LAYERS", 3))
DROPOUT = float(os.environ.get("GRAPHMIND_DROPOUT", 0.2))
PRETRAIN_EPOCHS = int(os.environ.get("GRAPHMIND_PRETRAIN_EPOCHS", 25))
FINETUNE_EPOCHS = int(os.environ.get("GRAPHMIND_FINETUNE_EPOCHS", 55))
N_PRETRAIN_EPISODES = int(os.environ.get("GRAPHMIND_N_PRETRAIN_EPISODES", 400))
POS_WEIGHT = float(os.environ.get("GRAPHMIND_POS_WEIGHT", 8.0))
BATCH_SIZE_PRETRAIN = int(os.environ.get("GRAPHMIND_BATCH_PRETRAIN", 32))
BATCH_SIZE_FINETUNE = int(os.environ.get("GRAPHMIND_BATCH_FINETUNE", 16))

# ---- Ablation ----
ABLATION_SEEDS = [int(s) for s in os.environ.get("GRAPHMIND_ABLATION_SEEDS", "1,2,3").split(",")]
ARCHITECTURES = ["GCN", "GraphSAGE", "GAT"]

# ---- Retrieval ----
TOP_K = int(os.environ.get("GRAPHMIND_TOP_K", 6))

# ---- Real HotpotQA (optional, network permitting) ----
HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HOTPOT_LOCAL_PATH = os.environ.get("GRAPHMIND_HOTPOT_PATH", "data/hotpot_dev_sample.json")
HOTPOT_SAMPLE_SIZE = int(os.environ.get("GRAPHMIND_HOTPOT_SAMPLE_SIZE", 50))

# ---- LLM generation ----
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")  # "ollama" or "anthropic"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ---- Output paths ----
RESULTS_DIR = os.environ.get("GRAPHMIND_RESULTS_DIR", "results")
METRICS_OUTPUT_PATH = os.path.join(RESULTS_DIR, "metrics.json")
