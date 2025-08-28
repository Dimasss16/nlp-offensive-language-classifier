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

# Model storage and loading

As a result of our baseline analysis, we saved two models: majority class and tf-idf + logreg. 

## Majority baseline

Location: `models/majority/`

Files:

- `majority.json` contains the majority class (always predicts class 1 -- 'offensive')
- `metrics.txt` - contains performance metrics


How to load:

```
import json
with open('models/majority/majority.json', 'r') as f:
    model_info = json.load(f)
majority_class = model_info['majority_class']
```

## TF-IDF + Logistic Regression

Location: `models/logreg/`

Files:

- `tfidf.joblib` - TF-IDF vectorizer (saved as object, *not dictionary*)
- `logreg.joblib` - Logistic Regression classifier (saved as object, *not dictionary*)

The above means it's improssible to load these with `.get()` method

Here is how to load:
```
import joblib

vectorizer = joblib.load('models/logreg/tfidf.joblib')
classifier = joblib.load('models/logreg/logreg.joblib')

text = "some text here"
text_vector = vectorizer.transform([text])
prediction = classifier.predict(text_vector)[0]
```

## Prediction file formats

### Majority baseline predictions
- **Columns**: `text`, `label`, `clean_text`, `prediction`
- `label`: True label (0/1/2)
- `prediction`: Always 1 (offensive)

### TF-IDF Predictions  
- **Columns**: `text`, `true_label`, `predicted_label`
- `true_label`: Actual class (0/1/2)
- `predicted_label`: Model prediction (0/1/2)

**Note**: We should unify the colum naming later. For now:
- Majority uses: `label`/`prediction`
- TF-IDF uses: `true_label`/`predicted_label`

## Class mapping
- 0: `hate_speech`
- 1: `offensive`
- 2: `neither`



