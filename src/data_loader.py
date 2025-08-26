import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import config


def download_data():
    if os.path.exists(config.RAW_DATA_PATH):
        print(f"Loading existing data from {config.RAW_DATA_PATH}")
        return pd.read_csv(config.RAW_DATA_PATH)

    print("Downloading dataset from HuggingFace...")
    dataset = load_dataset(config.DATASET_NAME)
    df = pd.DataFrame(dataset['train'])

    os.makedirs('data/raw', exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"Saved {len(df)} samples to {config.RAW_DATA_PATH}")

    return df


def create_splits(df):
    df = df.drop_duplicates(subset=['tweet']).reset_index(drop=True)
    print(f"After removing duplicates: {len(df)} samples")

    X = df['tweet'].values
    y = df['class'].values

    # First split: 85/15 (train+val / test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    # Second split: 70/15 of total
    val_size = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=config.RANDOM_STATE, stratify=y_temp
    )

    print(f"Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Save splits
    os.makedirs('data/splits', exist_ok=True)

    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    val_df = pd.DataFrame({'text': X_val, 'label': y_val})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})

    train_df.to_csv(config.TRAIN_PATH, index=False)
    val_df.to_csv(config.VAL_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)

    print(f"Splits saved to data/splits/")

    return train_df, val_df, test_df


if __name__ == "__main__":
    df = download_data()

    print("Class distribution:")
    for class_id, count in df['class'].value_counts().items():
        print(f"  {config.CLASS_NAMES[class_id]}: {count} ({100 * count / len(df):.1f}%)")

    train_df, val_df, test_df = create_splits(df)
