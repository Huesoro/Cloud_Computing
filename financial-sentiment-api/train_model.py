# train_model.py

import os
import pandas as pd
import kagglehub
import torch

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"

MODEL_DIR = "financial_sentiment_model"


# ==========================================
# DATASET CLASS
# ==========================================

class FinancialDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128
        )

        self.labels = labels

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item

    def __len__(self):

        return len(self.labels)


# ==========================================
# METRICS
# ==========================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = predictions.argmax(axis=1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy
    }


# ==========================================
# TRAIN MODEL
# ==========================================

def deploy_model():

    # --------------------------------------
    # 1. Download dataset
    # --------------------------------------

    print("Downloading dataset...")

    path = kagglehub.dataset_download(
        "sbhatti/financial-sentiment-analysis"
    )

    csv_path = os.path.join(
        path,
        "data.csv"
    )

    # --------------------------------------
    # 2. Load dataset
    # --------------------------------------

    data = pd.read_csv(csv_path)

    # Normalize column names
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )

    print("\nDataset loaded!")

    print(data.head())

    print("\nColumns:")
    print(data.columns.tolist())

    # --------------------------------------
    # 3. Clean data
    # --------------------------------------

    data = data.dropna(
        subset=[
            "sentence",
            "sentiment"
        ]
    ).copy()

    data["sentence"] = (
        data["sentence"]
        .astype(str)
        .str.strip()
    )

    data["sentiment"] = (
        data["sentiment"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # --------------------------------------
    # 4. Convert labels to numbers
    # --------------------------------------

    label2id = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    id2label = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    data["label"] = (
        data["sentiment"]
        .map(label2id)
    )

    data = data.dropna(
        subset=["label"]
    ).copy()

    data["label"] = (
        data["label"]
        .astype(int)
    )

    # --------------------------------------
    # 5. Original distribution
    # --------------------------------------

    print("\nOriginal class distribution:")

    print(
        data["sentiment"].value_counts()
    )

    # --------------------------------------
    # 6. Split BEFORE oversampling
    # --------------------------------------
    #
    # IMPORTANT:
    # The test set stays untouched.
    #

    train_data, test_data = train_test_split(

        data,

        test_size=0.20,

        random_state=42,

        stratify=data["label"]

    )

    print("\nOriginal training distribution:")

    print(
        train_data["sentiment"].value_counts()
    )

    print("\nTest distribution:")

    print(
        test_data["sentiment"].value_counts()
    )

    # --------------------------------------
    # 7. Oversample minority classes
    # --------------------------------------

    max_samples = (
        train_data["sentiment"]
        .value_counts()
        .max()
    )

    balanced_parts = []

    for sentiment in [
        "negative",
        "neutral",
        "positive"
    ]:

        class_data = train_data[
            train_data["sentiment"] == sentiment
        ]

        # Sample with replacement until
        # every class has the same size

        balanced_class = class_data.sample(

            n=max_samples,

            replace=True,

            random_state=42

        )

        balanced_parts.append(
            balanced_class
        )

    balanced_train_data = pd.concat(
        balanced_parts,
        ignore_index=True
    )

    # Shuffle balanced dataset

    balanced_train_data = (
        balanced_train_data
        .sample(
            frac=1,
            random_state=42
        )
        .reset_index(drop=True)
    )

    print("\nBalanced training distribution:")

    print(
        balanced_train_data["sentiment"]
        .value_counts()
    )

    # --------------------------------------
    # 8. Prepare X and y
    # --------------------------------------

    train_texts = (
        balanced_train_data["sentence"]
        .tolist()
    )

    train_labels = (
        balanced_train_data["label"]
        .tolist()
    )

    test_texts = (
        test_data["sentence"]
        .tolist()
    )

    test_labels = (
        test_data["label"]
        .tolist()
    )

    print(
        "\nTraining samples after balancing:",
        len(train_texts)
    )

    print(
        "Testing samples:",
        len(test_texts)
    )

    # --------------------------------------
    # 9. Load tokenizer
    # --------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------
    # 10. Create datasets
    # --------------------------------------

    train_dataset = FinancialDataset(

        train_texts,

        train_labels,

        tokenizer

    )

    test_dataset = FinancialDataset(

        test_texts,

        test_labels,

        tokenizer

    )

    # --------------------------------------
    # 11. Load model
    # --------------------------------------

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(

        MODEL_NAME,

        num_labels=3,

        id2label=id2label,

        label2id=label2id

    )

    # --------------------------------------
    # 12. Training configuration
    # --------------------------------------

    training_args = TrainingArguments(

        output_dir="./training_output",

        num_train_epochs=4,

        per_device_train_batch_size=16,

        per_device_eval_batch_size=16,

        learning_rate=3e-5,

        weight_decay=0.01,

        logging_steps=50,

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="accuracy",

        greater_is_better=True

    )

    # --------------------------------------
    # 13. Trainer
    # --------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=test_dataset,

        compute_metrics=compute_metrics

    )

    # --------------------------------------
    # 14. Train
    # --------------------------------------

    print("\n================================")
    print("TRAINING MODEL")
    print("================================")

    trainer.train()

    # --------------------------------------
    # 15. Evaluate
    # --------------------------------------

    print("\n================================")
    print("EVALUATING MODEL")
    print("================================")

    evaluation = trainer.evaluate()

    print("\nEvaluation results:")

    print(
        "Accuracy:",
        round(
            evaluation["eval_accuracy"],
            4
        )
    )

    # --------------------------------------
    # 16. Classification report
    # --------------------------------------

    print("\nClassification report:")

    predictions = trainer.predict(
        test_dataset
    )

    predicted_labels = (
        predictions.predictions
        .argmax(axis=1)
    )

    print(
        classification_report(

            test_labels,

            predicted_labels,

            target_names=[
                "negative",
                "neutral",
                "positive"
            ],

            zero_division=0

        )
    )

    # --------------------------------------
    # 17. Save model
    # --------------------------------------

    print("\nSaving model...")

    model.save_pretrained(
        MODEL_DIR
    )

    tokenizer.save_pretrained(
        MODEL_DIR
    )

    print(
        "\n================================"
    )

    print(
        "MODEL SAVED SUCCESSFULLY"
    )

    print(
        MODEL_DIR
    )

    print(
        "================================"
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    deploy_model()