"""
Configuration file - all constants in one place
"""

# Dataset
DATASET_NAME = 'tdavidson/hate_speech_offensive'

# Paths
RAW_DATA_PATH = 'data/raw/hate_speech.csv'
TRAIN_PATH = 'data/splits/train.csv'
VAL_PATH = 'data/splits/val.csv'
TEST_PATH = 'data/splits/test.csv'
# Preprocessed paths (using light preprocessing for TF-IDF)
PROCESSED_TRAIN_PATH = 'data/processed_light/train.csv'
PROCESSED_VAL_PATH = 'data/processed_light/val.csv'
PROCESSED_TEST_PATH = 'data/processed_light/test.csv'

# BERT preprocessing paths (minimal cleaning, preserves case)
BERT_TRAIN_PATH = 'data/processed_bert/train.csv'
BERT_VAL_PATH = 'data/processed_bert/val.csv'
BERT_TEST_PATH = 'data/processed_bert/test.csv'

# Model paths
TFIDF_MODEL_PATH = 'models/tfidf_logreg.pkl'

# Results paths
GROUND_ZERO_RESULTS = 'results/ground_zero_results.txt'
BASELINE_RESULTS = 'results/baseline_results.txt'

# Class names
CLASS_NAMES = {
    0: 'hate_speech',
    1: 'offensive',
    2: 'neither'
}

# Split ratios
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# TF-IDF parameters
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 3)