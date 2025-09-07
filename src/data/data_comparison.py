import pandas as pd

#data loader
def load_data(original_path, augmented_path):
    original_df = pd.read_csv(original_path)
    augmented_df = pd.read_csv(augmented_path)
    return original_df, augmented_df

#get distribution & class counts
def analyze_dataset(df, label_map={0: "Hate Speech", 1: "Offensive", 2: "Neither"}):
    print("\nClass Distribution (Raw Data):")
    print(df['label'].value_counts(normalize=True).to_string())

    class_counts = df['label'].value_counts()
    print(f"\nNumber of samples per class:")
    for label, count in class_counts.items():
        print(f"{label_map.get(label, label)}: {count}")

def compare_datasets(original_df, augmented_df, label_map={0: "Hate Speech", 1: "Offensive", 2: "Neither"}):
    print("\n--- Comparing Original and Augmented Datasets ---")

    #original dataset analysis
    print("\nOriginal Dataset Analysis:")
    analyze_dataset(original_df, label_map)

    #augmented dataset analysis
    print("\nAugmented Dataset Analysis:")
    analyze_dataset(augmented_df, label_map)

    #compare distributions (before vs after)
    original_class_counts = original_df['label'].value_counts()
    augmented_class_counts = augmented_df['label'].value_counts()

    print("\nClass distribution difference (Original vs Augmented):")
    for label in original_class_counts.index:
        original_count = original_class_counts.get(label, 0)
        augmented_count = augmented_class_counts.get(label, 0)
        difference = augmented_count - original_count
        print(f"{label_map.get(label, label)}: {original_count} -> {augmented_count} (Difference: {difference})")


def main():
    
    original_path = "src/data/processed_bert/train.csv"
    augmented_path = "src/data/processed_bert/train_augmented.csv"

    #öoad the datasets
    original_df, augmented_df = load_data(original_path, augmented_path)

    #compare the datasets
    compare_datasets(original_df, augmented_df)
    

if __name__ == "__main__":
    main()