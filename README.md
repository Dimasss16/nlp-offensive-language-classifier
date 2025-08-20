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





