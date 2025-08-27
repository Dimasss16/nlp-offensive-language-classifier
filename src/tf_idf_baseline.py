import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import os

# prep the directories
PRED_DIR = "predictions/logreg/"
MODEL_DIR = "models/logreg"
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def get_data(dataframe):
    X = dataframe["clean_text"]
    y = dataframe["label"]
    return X, y

def save_predictions(X, y, y_pred, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pd.DataFrame({
    "text": pd.Series(X).reset_index(drop=True),
    "true_label": pd.Series(y).reset_index(drop=True),
    "predicted_label": pd.Series(y_pred).reset_index(drop=True)
    })

    df.to_csv(out_path, index=False)
    

# load the files
train = pd.read_csv("src\\data\\processed_light\\train.csv")
val = pd.read_csv("src\\data\\processed_light\\val.csv")
test= pd.read_csv("src\\data\\processed_light\\test.csv")

# set up the data
X_train, y_train = get_data(train)
X_val, y_val = get_data(val)
X_test, y_test = get_data(test)

#vectorising 
td = TfidfVectorizer()
X_train_td = td.fit_transform(X_train)
X_val_td = td.transform(X_val)
X_test_td = td.transform(X_test)

#train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_td, y_train)

#saving the model and vectorizer
joblib.dump(model, os.path.join(MODEL_DIR, "logreg.joblib"))
joblib.dump(td, os.path.join(MODEL_DIR, "tfidf.joblib"))
## joblib.load("model.joblib") <- to load the model

#test on the validation set
y_val_pred = model.predict(X_val_td)
save_predictions(X_val, y_val, y_val_pred, "predictions/logreg/val_predictions.csv")

print('\nValidation Accuracy: ', accuracy_score(y_val, y_val_pred))
print('\nValidation Classification Report')
print('\n', classification_report(y_val, y_val_pred))


y_test_pred = model.predict(X_test_td)
save_predictions(X_test, y_test, y_test_pred, "predictions/logreg/test_predictions.csv")

metrics_path = os.path.join(PRED_DIR, "metrics.txt")
with open(metrics_path, "w") as f: 
    f.write(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")
    f.write(f"\nTest Macro-F1: {f1_score(y_test, y_test_pred, average='macro')}")
    f.write(f"\nTest Classification Report")
    f.write(f"\n {classification_report(y_test, y_test_pred)}")