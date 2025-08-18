# Offensive Language Classifier

We train on 25,000 annotated tweets from Hugging Face (hate, offensive, neither). Start with a TF-IDF + Logistic Regression baseline, then fine-tune DistilBERT. We clean the text and apply light augmentation. For evaluation we use class-balanced F1/precision/recall with per-class reports. 
