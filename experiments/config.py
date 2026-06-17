"""CD4Code Experiment Configuration.

All secrets and proxy settings are read from environment variables.
Never hard-code API keys in this file.
"""
import os

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# Proxy configuration (from environment variable)
HTTP_PROXY = os.environ.get("HTTP_PROXY", os.environ.get("http_proxy", ""))

# CD4Code tier parameters
TIER1_CONFIDENCE_THRESHOLD = 0.6       # Token-level proofreading
TIER2_LINTER = "pylint"                # Static analysis tool
TIER2_TYPE_CHECKER = "mypy"            # Type checker
TIER3_MAX_RETRIES = 3                  # Max regeneration attempts
TIER4_DEFECT_THRESHOLD = 0.4           # Global defect density threshold
TIER4_CONSERVATIVE_TEMP = 0.3          # Conservative temperature
TIER4_CONSERVATIVE_TOPP = 0.85         # Conservative top-p

# Threshold sensitivity sweep (Experiment 4)
THRESHOLD_SWEEP_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
T1_MIN_LENGTH_VALUES = [1, 3, 5, 10, 20]
T1_REPETITION_WINDOW_VALUES = [3, 5, 8, 10, 15, 20]

# Generation defaults
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 512

# Tier4 stress test configurations
STRESS_TEMPERATURE = 1.5               # High temperature to elevate defect rate
STRESS_TOP_P = 1.0                     # Full nucleus sampling
STRESS_HARD_PROBLEM_COUNT = 50         # Number of hardest problems to select
STRESS_PERTURB_RATE = 0.3              # Fraction of prompt tokens to perturb
STRESS_WEAKER_TEMP = 1.2               # Intermediate temperature for weaker generation
STRESS_WEAKER_TOPP = 0.99              # Intermediate top-p for weaker generation
STRESS_WEAKER_MAX_TOKENS = 150         # Reduced max tokens to simulate weaker model

# MBPP configuration
MBPP_DEFAULT_SAMPLES = 50              # Default sample size for MBPP
MBPP_RANDOM_SEED = 42                  # Random seed for MBPP sampling

# API pricing (USD per 1M tokens, DeepSeek-v4-flash)
PRICE_INPUT_PER_1M = 0.27              # Input token price
PRICE_OUTPUT_PER_1M = 1.10             # Output token price

# Bootstrap settings
BOOTSTRAP_SAMPLES = 10000              # Bootstrap resamples for CI
BOOTSTRAP_CONFIDENCE = 0.95            # Confidence level

# Paths
HUMANEVAL_PATH = "data/HumanEval.jsonl"
MBPP_PATH = "data/mbpp.jsonl"
RESULTS_DIR = "results"
FIGURES_DIR = "../paper/figures"
