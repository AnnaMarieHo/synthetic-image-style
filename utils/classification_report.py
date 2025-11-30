import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# Load JSONL into DataFrame
path = "openfake-annotation/datasets/combined/llm_training_data.jsonl"
records = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
df = pd.DataFrame(records)[:19932]



le = LabelEncoder()
df["true_label"] = le.fit_transform(df["true_label"])
df["prediction"] = le.transform(df["prediction"])




cm = confusion_matrix(df["true_label"], df["prediction"])
report = classification_report(df["true_label"], df["prediction"], target_names=le.classes_)

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
