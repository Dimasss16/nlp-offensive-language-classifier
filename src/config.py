DATASET_NAME = 'tdavidson/hate_speech_offensive'

# Paths
RAW_DATA_PATH = 'src/data/raw/hate_speech.csv'
TRAIN_PATH = 'src/data/splits/train.csv'
VAL_PATH = 'src/data/splits/val.csv'
TEST_PATH = 'src/data/splits/test.csv'
# Preprocessed paths (using light preprocessing for TF-IDF)
PROCESSED_TRAIN_PATH = 'src/data/processed_light/train.csv'
PROCESSED_VAL_PATH = 'src/data/processed_light/val.csv'
PROCESSED_TEST_PATH = 'src/data/processed_light/test.csv'

# BERT preprocessing paths (minimal cleaning, preserves case)
BERT_TRAIN_PATH = 'src/data/processed_bert/train.csv'
BERT_VAL_PATH = 'src/data/processed_bert/val.csv'
BERT_TEST_PATH = 'src/data/processed_bert/test.csv'

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