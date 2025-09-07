#script for data augmentation (synonym replacement, character level noise)

import pandas as pd
import random
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nlpaug.augmenter.char as nac

#reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")

#loading list of profanity words
def load_profanity_list(path="src/data/bad-words.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return set(w.strip().lower() for w in f if w.strip())


STOPWORDS = set(stopwords.words("english"))
PROFANITY = load_profanity_list()

#character noise augmenter
char_aug = nac.KeyboardAug(aug_char_min=1, aug_char_max=10, aug_word_p=0.1)

#synonym replacement function
def synonym_replacement(text, replace_prob=0.1):
    """Replace common words with synonyms (probability-based)."""
    words = text.split()
    new_words = []
    for w in words:
        if random.random() < replace_prob:
            synonyms = wordnet.synsets(w)
            if synonyms:
                lemmas = [lemma.name().replace("_", " ") for syn in synonyms for lemma in syn.lemmas()]
                lemmas = [l for l in lemmas if l.lower() != w.lower()]
                if lemmas:
                    new_words.append(random.choice(lemmas))
                    continue
        new_words.append(w)
    return " ".join(new_words)


#augmentation pipeline
def augment_data(train_path, output_path):
    df = pd.read_csv(train_path)

    hate_df = df[df["label"] == 0]
    offensive_df = df[df["label"] == 1]
    neither_df = df[df["label"] == 2]

    augmented = []

    #hate speech → 2 augmentations (1 synonym, 1 char noise)
    for _, row in hate_df.iterrows():
        text = row["clean_text"]

        aug1 = synonym_replacement(text)
        aug2 = char_aug.augment(text)

        augmented.append({"text": aug1, "label": 0, "clean_text": aug1})
        augmented.append({"text": aug2, "label": 0, "clean_text": aug2})

    #neither → 50% synonym, 50% char noise
    for _, row in neither_df.iterrows():
        text = row["clean_text"]
        if random.random() < 0.5:
            aug_text = synonym_replacement(text)
        else:
            aug_text = char_aug.augment(text)
        augmented.append({"text": aug_text, "label": 2, "clean_text": aug_text})

    #combine original + augmented
    aug_df = pd.DataFrame(augmented)
    final_df = pd.concat([df, aug_df], ignore_index=True)

    final_df.to_csv(output_path, index=False)
    print(f"Saved augmented training set to {output_path}")
    print("New class distribution:")
    print(final_df["label"].value_counts(normalize=True))

if __name__ == "__main__":
    train_path = "src/data/processed_bert/train.csv"
    output_path = "src/data/processed_bert/train_augmented.csv"
    augment_data(train_path, output_path)