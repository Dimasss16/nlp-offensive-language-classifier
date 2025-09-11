import pandas as pd
import random
import nlpaug.augmenter.char as nac
import numpy as np
import nltk
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split

#reproductibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

#nltk setup
nltk.download("wordnet")

#character augmentation 
char_aug = nac.RandomCharAug(aug_char_min=1, aug_char_max=10, aug_char_p=0.1)
def char_augment(text): 
    return char_aug.augment(text)

def synonym_replacement(text, replace_prob = 0.1):
    """Replace common words with synonyms (probability-based)."""
    words = text.split()
    new_words = []
    for w in words: 
        if random.random() < replace_prob:
            synonyms = wordnet.synsets(w)
            if synonyms: #if there are synonyms 
                lemmas = [lemma.name().replace("_", " ") for syn in synonyms for lemma in syn.lemmas()]
                lemmas = [l for l in lemmas if l.lower() != w.lower()]
                if lemmas: 
                    chosen = random.choice(lemmas)
                    # print(f"Replacing {w} with {chosen}") - debug
                    new_words.append(chosen)
                    continue
        new_words.append(w)
    return " ".join(new_words)

def do_perturbation(test_path, output_path):
    # get the df
    df = pd.read_csv(test_path)

    # get 100 random tweets from the data
    random_texts = df["clean_text"].sample(n=100, random_state=42)
    char_aug_texts, syn_aug_texts = train_test_split(
        random_texts, test_size=0.5, random_state=42
    )

    char_aug_texts = char_aug_texts.apply(char_augment)
    print(f"Example character augmented texts: {char_aug_texts.head()}")
    syn_aug_texts = syn_aug_texts.apply(synonym_replacement)   
    print(f"Example synonym augmented texts: {syn_aug_texts.head()}")


    df.loc[char_aug_texts.index, "clean_text"] = char_aug_texts
    df.loc[syn_aug_texts.index, "clean_text"] = syn_aug_texts    

    df.to_csv(output_path, index=False)
    print(f"Saved test set with perturbations to: {output_path}")


if __name__ == "__main__":
    test_path = "src/data/processed_bert/test.csv"
    output_path = "src/data/processed_bert/test_perturbation.csv"
    do_perturbation(test_path, output_path)

                 