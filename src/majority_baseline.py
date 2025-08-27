import pandas as pd

#save file paths
train_path = "src/data/processed_light/train.csv"
val_path   = "src/data/processed_light/val.csv"
test_path  = "src/data/processed_light/test.csv"

#identify majority class
train_df = pd.read_csv(train_path)
majority_class = train_df["label"].value_counts().idxmax()
print(f"Majority class (training set): {majority_class}")

#generate predictions
def make_predictions(input_path, output_path, majority_class):
    df = pd.read_csv(input_path)
    df["prediction"] = majority_class
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return df

val_out  = "predictions/baseline_val_predictions.csv"
test_out = "predictions/baseline_test_predictions.csv"

val_df = make_predictions(val_path, val_out, majority_class)
test_df = make_predictions(test_path, test_out, majority_class)

#eval on val data
val_acc = (val_df["label"] == val_df["prediction"]).mean()
print(f"Baseline validation accuracy: {val_acc:.2%}")


