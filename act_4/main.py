from fastapi import FastAPI
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch
import torch.nn.functional as F


# -----------------------------------
# Configuración
# -----------------------------------

MODEL_DIR = "financial_sentiment_model"


# -----------------------------------
# Crear API
# -----------------------------------

app = FastAPI(
    title="Financial Sentiment Analysis API",
    description="API for financial sentiment analysis using a small BERT transformer",
    version="1.0.0"
)


# -----------------------------------
# Cargar modelo
# -----------------------------------

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()

print("MODEL FOUND!")


# -----------------------------------
# Input
# -----------------------------------

class InputData(BaseModel):
    text: str


# -----------------------------------
# Home
# -----------------------------------

@app.get("/")
def home():

    return {
        "message": "Financial Sentiment Analysis API is running"
    }


# -----------------------------------
# Prediction
# -----------------------------------

@app.post("/predict")
def predict(data: InputData):

    # Tokenizar texto
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )


    # Obtener predicción
    with torch.no_grad():

        outputs = model(**inputs)


    # Convertir logits a probabilidades
    probabilities = F.softmax(
        outputs.logits,
        dim=1
    )


    # Obtener clase con mayor probabilidad
    prediction_id = torch.argmax(
        probabilities,
        dim=1
    ).item()


    # Obtener confianza
    confidence = probabilities[
        0,
        prediction_id
    ].item()


    # Convertir ID a nombre
    sentiment = model.config.id2label[
        prediction_id
    ]


    return {

        "text": data.text,

        "sentiment": sentiment,

        "confidence": float(confidence)

    }