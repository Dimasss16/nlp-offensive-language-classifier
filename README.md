# Offensive Language Classifier

We train on 25,000 annotated tweets from Hugging Face (hate, offensive, neither). Start with a TF-IDF + Logistic Regression baseline, then fine-tune DistilBERT. We clean the text and apply light augmentation. For evaluation we use class-balanced F1/precision/recall with per-class reports. 

# Some notes on setup and usage

```
pyenv local 3.11.9 # most stable

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

source .venv/bin/activate
```

# Git

```
git pull
git add --all
git commit -m "Commit message"
git pull (just in case)
git push -u origin main
```

# Some notes on data preprocessing

During preprocessing we created two cleaned versions of the train/val/test splits. These are going to be used for tf idf and bert respectively. We just removed **true noise** (links, HTML artifacts, inconsistent spacing) but kept the stop words because they are important for sentiment analysis. 


## Inputs & outputs

Here is how the data was handled:

* Inputs: `src/data/splits/{train,val,test}.csv` with columns `text,label`
* Outputs:

  * `src/data/processed_light/{train,val,test}.csv` for TF-IDF + Logistic Regression
  * `src/data/processed_bert/{train,val,test}.csv` for BERT
* Each output file keeps the original `text`, adds `clean_text`, and preserves `label`.

  Here are the cleaning details:

## Light cleaning (for TF-IDF)

* Replace `@username` with `@user` (keeps the mention signal)
* Replace URLs with `[URL]`, then lowercase (so it appears as `[url]`)
* Strip HTML entities
* Lowercase all text
* Collapse repeated whitespace
* We do not remove stopwords, pronouns, negations, contractions, or punctuation.

## Minimal cleaning (for BERT)

* Replace `@username` with `@user`
* Replace URLs with `[URL]` (case is preserved)
* Strip HTML entities
* Collapse repeated whitespace
* We keep the original casing and punctuation for bert.




