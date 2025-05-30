# analyzer.py

import nltk
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import torch

from suicide_phrases import contains_suicidal_phrase  



# Descargar recursos necesarios para NLTK
nltk.download('vader_lexicon')

# Inicializar el analizador de sentimientos de NLTK (VADER)
sentiment_analyzer = SentimentIntensityAnalyzer()

device = 0 if torch.cuda.is_available() else -1

# Inicializar el modelo de clasificación de emociones
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device,
    return_all_scores=False
)

# Inicializar el modelo de clasificación de intenciones
intent_classifier = pipeline(
    "text-classification",
    model="BerserkerMother/all-MiniLM-L6-v2-intent-classifier",
    device=device,
    return_all_scores=False
)

def analyze_text(text):
    """
    Analiza el texto proporcionado y devuelve el sentimiento, la emoción y la intención.
    """
    # Análisis de sentimiento
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment = 'positive'
    elif compound_score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    # Análisis de emoción
    try:
        emotion_result = emotion_classifier(text)
        emotion = emotion_result[0]['label']
    except Exception as e:
        emotion = 'unknown'

    # Análisis de intención
    try:
        intent_result = intent_classifier(text)
        intent = intent_result[0]['label']
    except Exception as e:
        intent = 'unknown'

    suicidal_flag = contains_suicidal_phrase(text)

    return {
        'sentiment': sentiment,
        'emotion': emotion,
        'intent': intent,
        'suicidal_phrase': suicidal_flag
    }
