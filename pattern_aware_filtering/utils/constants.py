import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths for CCNet data
CCWET_PATH = "data/ccnet/wet" # <<< /shared/erc/dataset/CC_WET

# baseline directories for English and Korean data
EN_CCWET_PATH = "data/ccwet/en" # /shared/erc/lab08/jia/workspace 
KO_CCWET_PATH = "data/ccwet/ko" # /shared/erc/lab08/jia/workspace

# Ablation data directory path
EN_EXTRACTED_PATH = "data/extracted/en"
KO_EXTRACTED_PATH = "data/extracted/ko"

# Intermediate hash directory path
EN_HASH_DIR = "data/hash/en"
KO_HASH_DIR = "data/hash/ko"

KO_SHARDED_HASH_DIR = os.path.join(KO_HASH_DIR, "sharded")
KO_MERGED_HASH_DIR = os.path.join(KO_HASH_DIR, "merged")
EN_SHARDED_HASH_DIR = os.path.join(EN_HASH_DIR, "sharded")
EN_MERGED_HASH_DIR = os.path.join(EN_HASH_DIR, "merged")


# CCWET Path file directory
CCWET_PATHFILE_DIR = "cache/wet_paths"

