import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

train = pd.read_csv("src\\data\\processed_light\\train.csv")
val = pd.read_csv("src\\data\\processed_light\\val.csv")
test= pd.read_csv("src\\data\\processed_light\\test.csv")

X_train = train["clean_text"]
y_train = train["label"]

X_val = val["clean_text"]
y_val = val["label"]

X_test = test["clean_text"]
y_test = test["label"]


#vectorising 
td = TfidfVectorizer()
X_train_td = td.fit_transform(X_train)
X_val_td = td.transform(X_val)
X_test_td = td.transform(X_test)

#train the model
model = LogisticRegression()
model.fit(X_train_td, y_train)

#saving the model and the vectorizer
save_dir = "models/logreg"
os.makedirs(save_dir, exist_ok=True)
joblib.dump(model, os.path.join(save_dir, "logreg.pkl"))
joblib.dump(td, os.path.join(save_dir, "tfidf.pkl"))

#test on the validation set
y_val_pred = model.predict(X_val_td)

print('\nValidation Accuracy: ', accuracy_score(y_val, y_val_pred))
print('\nValidation Classification Report')
print('======================================================')
print('\n', classification_report(y_val, y_val_pred))


y_test_pred = model.predict(X_test_td)

print('\nTest Accuracy: ', accuracy_score(y_test, y_test_pred))
print('\nTest Classification Report')
print('======================================================')
print('\n', classification_report(y_test, y_test_pred))