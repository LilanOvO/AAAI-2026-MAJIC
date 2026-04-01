# Configuration Template for MAJIC
# Copy this file to config.py and fill in your API keys

# OpenAI API Configuration
API_SECRET_KEY = "your-api-key-here"
BASE_URL = "https://api.openai.com/v1"  # or your custom endpoint

# Model Configuration
ATTACKER_MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
VICTIM_MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"

# Attack Configuration
ATTACK_TYPE = "gpt-4o"  # Options: "local", "gpt-4o", "gpt-3.5-turbo", "claude-3-5-sonnet", etc.
JUDGE_TYPE = "gpt"  # Options: "gpt", "llama2", "rule"

# Markov Chain Parameters
CHAIN_COUNT = 10        # Number of attack chains per query
CHAIN_LENGTH = 3        # Max optimization steps per chain
INIT_QNUM = 1          # Queries for initial method
CHAIN_QNUM = 1         # Queries per optimization step

# Learning Parameters
GAMMA = 0.5            # Discount factor for Q-learning
ALPHA = 0.1            # Learning rate
BETA = 0.01            # Exploration rate
TEMPERATURE = 0.15     # Softmax temperature

# Data Paths
DATA_PATH = "data/harmful_behaviors_50.json"
RESULTS_PATH = "results/output.json"
MATRIX_PATH = "markov_methods/matrix.npy"
