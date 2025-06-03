"""
Sentiment Analyser module

Defines SentimentAnalyser:
 - loads tokenizer & model from Hugging Face
 - provides analyse(text: str) -> dict with sentiment scores
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from .config import MODEL_NAME, SCORE_COLUMNS

class SentimentAnalyser:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()

    def analyse(self, text: str) -> dict:
        """
        Analyse the given text and return sentiment scores.

        Args:
            text (str): Free-text input to analyse.

        Returns:
            dict: Mapping of score labels to float probabilities.
        """
        # Tokenize input and run model
        encoded = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded)
        scores = output.logits[0].detach().numpy()

        # Convert logits to probabilities
        probs = softmax(scores)

        # Map to keys
        return {label: float(probs[idx]) for idx, label in enumerate(SCORE_COLUMNS)}
