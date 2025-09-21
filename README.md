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

Here’s a drop-in **BERT training & results** section for your README, matching your tone and keeping it mostly plain text with two compact tables.

---

## BERT fine-tuning (clean vs. augmented)

We fine-tune a BERT classifier on the same tweet splits as the baselines, then repeat training on an **augmented** version of the training set to mitigate class imbalance.

### Data and splits
Train/val/test sizes: 17,347 / 3,718 / 3,718 (clean). After augmentation (+2× for *hate speech*, +1× for *neither*), train becomes 22,263 while val/test stay the same. Class shares shift from \~77/17/6 (*offensive/neither/hate*) to \~60/26/14.

### Class weighting
Weighted cross-entropy (scikit-learn “balanced” style) to counter imbalance:

* Clean train weights ≈ `[5.7766, 0.4305, 1.9843]` for `[hate, offensive, neither]`.
* Augmented train weights ≈ `[2.4712, 0.5525, 1.2733]`.

### Training config
BERT encoder with a linear head, max length 128, AdamW (lr `3e-5`, wd `0.01`, warmup `10%`), 4 epochs, early stopping on val macro-F1 (patience 2), fp16 enabled. Per-device batch size 16 with grad-accum 2 → effective batch 32.

### What changes with augmentation
On the clean test set, the augmented model slightly improves overall scores and **trades a bit of hate-speech recall for higher precision**—raising hate-speech F1 overall. It also **improves offensive recall**. Under perturbation, **hate-speech recall for the augmented model is unchanged** (clean vs. perturbed), while its precision/F1 dip modestly.

### Overall test metrics (clean)

| Model    |  Accuracy |  Macro F1 | Weighted F1 |
| -------- | --------: | --------: | ----------: |
| BERT     |     0.902 |     0.772 |       0.907 |
| Aug-BERT | **0.911** | **0.783** |   **0.913** |

### Per-class (clean test)

**Original BERT**

| Class       | Precision | Recall |    F1 |
| ----------- | --------: | -----: | ----: |
| Hate speech |     0.414 |  0.570 | 0.479 |
| Offensive   |     0.953 |  0.929 | 0.941 |
| Neither     |     0.901 |  0.888 | 0.894 |

**Aug-BERT**

| Class       | Precision | Recall |    F1 |
| ----------- | --------: | -----: | ----: |
| Hate speech |     0.474 |  0.547 | 0.508 |
| Offensive   |     0.950 |  0.944 | 0.947 |
| Neither     |     0.903 |  0.883 | 0.893 |

Our main observations

* Hate speech: **precision went up from 0.414 to 0.474**, **recall decreased from 0.570 to 0.547**, **F1 climbed from 0.479 to 0.508**.
* Offensive: **recall increased from 0.929 to 0.944** (precision roughly unchanged).
* Overall we saw small but consistent gains in Accuracy and Macro F1.

### Perturbation
On 100 perturbed test samples (char noise / synonym swap), macro-F1 deltas are small (~−0.015 Aug-BERT, ~−0.005 BERT). For **Aug-BERT**, hate-speech **recall is identical** clean vs. perturbed (stayed at 0.547), while precision drops (appr −0.05) and F1 drops (appr −0.03).

Files:

* Metrics CSVs and predictions:

  * `results/bert_original/{clean,perturbed}/metrics.csv`
  * `results/bert_original/{clean,perturbed}/test_predictions.csv`
  * `results/bert_augmented/{clean,perturbed}/metrics.csv`
  * `results/bert_augmented/{clean,perturbed}/test_predictions.csv`
 
* Confusion matrices & per-class plots are saved alongside the metrics under `results/`.


## Class mapping
- 0: `hate_speech`
- 1: `offensive`
- 2: `neither`

# Repo structure
```
.
├── LICENSE
├── README.md
├── models
│   ├── bert_augmented
│   │   ├── best_model
│   │   │   ├── config.json
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.txt
│   │   ├── class_weights.json
│   │   ├── dataset_info.json
│   │   ├── training_config.json
│   │   └── training_history.json
│   ├── bert_original
│   │   ├── best_model
│   │   │   ├── config.json
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.txt
│   │   ├── class_weights.json
│   │   ├── dataset_info.json
│   │   ├── training_config.json
│   │   └── training_history.json
│   ├── logreg
│   │   ├── logreg.joblib
│   │   └── tfidf.joblib
│   └── majority
│       ├── majority.json
│       └── metrics.txt
├── notebooks
│   ├── bert_train_eval.ipynb
│   ├── bert_training.ipynb
│   ├── bert_training_fixed.ipynb
│   ├── eval_visualisation.ipynb
│   ├── perturbation_tests.ipynb
│   ├── stage1_evaluation.ipynb
│   └── {01_eda.ipynb}
├── predictions
│   ├── bert
│   │   ├── bert_augmented
│   │   │   ├── metrics.csv
│   │   │   ├── test_predictions.csv
│   │   │   └── val_predictions.csv
│   │   └── bert_original
│   │       ├── metrics.csv
│   │       ├── test_predictions.csv
│   │       └── val_predictions.csv
│   ├── logreg
│   │   ├── metrics.txt
│   │   ├── test_predictions.csv
│   │   └── val_predictions.csv
│   └── majority
│       ├── majority_baseline_test_predictions.csv
│       └── majority_baseline_val_predictions.csv
├── requirements.txt
├── results
│   ├── bert_augmented
│   │   ├── clean
│   │   │   └── test_predictions.csv
│   │   └── perturbed
│   │       └── test_predictions.csv
│   ├── bert_eval_confusion_matrices.png
│   ├── bert_original
│   │   ├── clean
│   │   │   └── test_predictions.csv
│   │   └── perturbed
│   │       └── test_predictions.csv
│   ├── general_evals.png
│   ├── per_class_comparison.png
│   ├── stage1_confusion_matrices.png
│   ├── stage1_overall_metrics.csv
│   └── stage1_per_class_comparison.png
├── run_phase1.py
└── src
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-311.pyc
    │   └── config.cpython-311.pyc
    ├── config.py
    ├── data
    │   ├── bad-words.txt
    │   ├── data_comparison.py
    │   ├── processed
    │   ├── processed_bert
    │   │   ├── test.csv
    │   │   ├── test_perturbation.csv
    │   │   ├── train.csv
    │   │   ├── train_augmented.csv
    │   │   └── val.csv
    │   ├── processed_light
    │   │   ├── test.csv
    │   │   ├── train.csv
    │   │   └── val.csv
    │   ├── raw
    │   │   └── hate_speech.csv
    │   └── splits
    │       ├── test.csv
    │       ├── train.csv
    │       └── val.csv
    ├── data_augmentation.py
    ├── data_loader.py
    ├── data_perturbation.py
    ├── majority_baseline.py
    ├── preprocessing.py
    └── tf_idf_baseline.py
```



