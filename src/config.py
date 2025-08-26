from pathlib import Path

DATASET_NAME = 'tdavidson/hate_speech_offensive'

# Base directory for data we create and use
BASE_DIR = Path('src/data')

RAW_DATA_PATH = BASE_DIR / 'raw/hate_speech.csv'
TRAIN_PATH = BASE_DIR / 'splits/train.csv'
VAL_PATH = BASE_DIR / 'splits/val.csv'
TEST_PATH = BASE_DIR / 'splits/test.csv'

# Preprocessed paths for tf-idf
PROCESSED_TRAIN_PATH = BASE_DIR / 'processed_light/train.csv'
PROCESSED_VAL_PATH = BASE_DIR / 'processed_light/val.csv'
PROCESSED_TEST_PATH = BASE_DIR / 'processed_light/test.csv'

# BERT preprocessing paths
BERT_TRAIN_PATH = BASE_DIR / 'processed_bert/train.csv'
BERT_VAL_PATH = BASE_DIR / 'processed_bert/val.csv'
BERT_TEST_PATH = BASE_DIR / 'processed_bert/test.csv'

CLASS_NAMES = {
    0: 'hate_speech',
    1: 'offensive',
    2: 'neither'
}

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42