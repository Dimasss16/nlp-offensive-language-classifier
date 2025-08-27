import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report

#save file paths
train_path = "src/data/processed_light/train.csv"
val_path   = "src/data/processed_light/val.csv"
test_path  = "src/data/processed_light/test.csv"

#identify majority class
train_df = pd.read_csv(train_path)
majority_class = train_df["label"].value_counts().idxmax()
print(f"Majority class (training set): {majority_class}")

#save model
model_path = "models/majority/majority.json"
with open(model_path, "w") as f:
    json.dump({"majority_class": int(majority_class)}, f)
print(f"Saved majority baseline model to {model_path}")


#generate predictions
def make_predictions(input_path, output_path, majority_class):
    df = pd.read_csv(input_path)
    df["prediction"] = majority_class
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return df

val_out  = "predictions/majority/majority_baseline_val_predictions.csv"
test_out = "predictions/majority/majority_baseline_test_predictions.csv"

val_df = make_predictions(val_path, val_out, majority_class)
test_df = make_predictions(test_path, test_out, majority_class)


#evaluation on validation data
y_true = val_df["label"]
y_pred = val_df["prediction"]

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
report   = classification_report(y_true, y_pred)

# Save metrics
metrics_path = "models/majority/metrics.txt"
with open(metrics_path, "w") as f:
    f.write(f"Majority class: {majority_class}\n")
    f.write(f"Validation accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1 score: {macro_f1:.4f}\n\n")
    f.write("Full classification report:\n")
    f.write(report)

print(f"Validation accuracy: {accuracy:.2%}")
print(f"Macro F1 score: {macro_f1:.4f}")
print(f"Saved metrics to {metrics_path}")
print("Finished.")