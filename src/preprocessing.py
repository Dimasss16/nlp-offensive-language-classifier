import re
import pandas as pd
import config


def remove_html_entities(text):
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    return text


def replace_mentions(text):
    text = re.sub(r'@[^\s]+', '@user', text)
    return text


def replace_urls(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '[URL]', text)

    text = re.sub(r'www\.[^\s]+', '[URL]', text)

    return text


def normalize_whitespace(text):
    text = ' '.join(text.split())
    return text.strip()


def light_preprocess(text):
    text = replace_mentions(text)
    text = replace_urls(text)
    text = remove_html_entities(text)

    text = text.lower()

    text = normalize_whitespace(text)

    return text


def minimal_preprocess(text):
    text = replace_mentions(text)
    text = replace_urls(text)
    text = remove_html_entities(text)

    text = normalize_whitespace(text)

    return text


def preprocess_splits():
    config.PROCESSED_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.BERT_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    for split_name in ['train', 'val', 'test']:
        print(f"Processing {split_name} set...")

        if split_name == 'train':
            input_path = config.TRAIN_PATH
        elif split_name == 'val':
            input_path = config.VAL_PATH
        else:
            input_path = config.TEST_PATH

        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} samples")

        # light preprocessing for tf-idf
        df_light = df.copy()
        df_light['clean_text'] = df_light['text'].apply(light_preprocess)

        # minimal preprocessing for bert
        df_bert = df.copy()
        df_bert['clean_text'] = df_bert['text'].apply(minimal_preprocess)

        if split_name == 'train':
            light_path = config.PROCESSED_TRAIN_PATH
            bert_path = config.BERT_TRAIN_PATH
        elif split_name == 'val':
            light_path = config.PROCESSED_VAL_PATH
            bert_path = config.BERT_VAL_PATH
        else:
            light_path = config.PROCESSED_TEST_PATH
            bert_path = config.BERT_TEST_PATH

        df_light.to_csv(light_path, index=False)
        df_bert.to_csv(bert_path, index=False)

        print(f"Saved light version to {light_path}")
        print(f"Saved BERT version to {bert_path}")

        if split_name == 'train':
            print("Examples of preprocessing:")
            for i in range(min(2, len(df))):
                print(f"[{i}] Original: {df.iloc[i]['text'][:80]}...")
                print(f"Light: {df_light.iloc[i]['clean_text'][:80]}...")
                print(f"BERT: {df_bert.iloc[i]['clean_text'][:80]}...")

    train_light = pd.read_csv(config.PROCESSED_TRAIN_PATH)
    train_bert = pd.read_csv(config.BERT_TRAIN_PATH)

    print("Let's compare what we preprocessed:")
    print()
    print(f"Light: avg {train_light['clean_text'].str.split().str.len().mean():.1f} words")
    print(f"BERT:  avg {train_bert['clean_text'].str.split().str.len().mean():.1f} words")


if __name__ == "__main__":
    preprocess_splits()