"""
Text Preprocessing Module

This module contains the TextPreprocessor class for both cleaning and advanced preprocessing of text for sentiment analysis.
Supports both minimal cleaning (for transformer models) and extensive preprocessing (for classical ML models).
"""

import re
import logging
from typing import List
from bs4 import BeautifulSoup
import contractions
import emoji
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.util import mark_negation
from sentiment_analysis.exceptions import InvalidPreprocessingModeError

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    A comprehensive text preprocessing class for sentiment analysis.
    
    Supports two modes:
    1. 'classical' - Extensive preprocessing for classical NLP
    2. 'transformer' - Minimal preprocessing for transformer models
    
    Features:
    - HTML tag removal
    - URL removal
    - Emoji handling (optional)
    - Contraction expansion (classical only)
    - Number replacement (classical only)
    - Tokenization (classical only)
    - Lemmatization with POS tagging (classical only)
    - Stopword removal (classical only)
    - Negation handling (classical only)
    """
    VALID_MODES = ['classical', 'transformer']
    def __init__(
        self, 
        mode: str = 'classical',
        use_negation: bool = True, 
        remove_stopwords: bool = True
    ):
        if mode not in self.VALID_MODES:
            raise InvalidPreprocessingModeError(mode, self.VALID_MODES)
        self.mode = mode
        self.use_negation = use_negation if mode == 'classical' else False
        self.remove_stopwords = remove_stopwords if mode == 'classical' else False
        logger.debug(f"Initialized TextPreprocessor (mode={mode})")
        if mode == 'classical':
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            self._download_nltk_data()
        else:
            self.lemmatizer = None
            self.stop_words = None
    def _download_nltk_data(self):
        required_data = [
            'stopwords',
            'wordnet',
            'omw-1.4',
            'punkt',
            'punkt_tab',
            'averaged_perceptron_tagger_eng'
        ]
        for dataset in required_data:
            try:
                nltk.data.find(f'corpora/{dataset}')
            except LookupError:
                nltk.download(dataset, quiet=True)
    @staticmethod
    def get_wordnet_pos(word: str) -> str:
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    def _remove_html_tags(self, text: str) -> str:
        # Fast path: if no HTML tags detected, return as-is
        if '<' not in text:
            return text
        return BeautifulSoup(text, "html.parser").get_text()
    def _remove_urls(self, text: str) -> str:
        return re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
    def _convert_emojis(self, text: str) -> str:
        return emoji.demojize(text)
    def _expand_contractions(self, text: str) -> str:
        return contractions.fix(text)
    def _replace_numbers(self, text: str) -> str:
        return re.sub(r'\d+', ' <num> ', text)
    def _remove_non_alpha(self, text: str) -> str:
        return re.sub(r'[^a-zA-Z\s<>]', '', text)
    def _tokenize(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        return [w for w in tokens if w.isalpha() or w == '<num>']
    def _apply_negation(self, tokens: List[str]) -> List[str]:
        return mark_negation(tokens)
    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        pos_tags = pos_tag(tokens)
        clean_tokens = []
        for word, _ in pos_tags:
            if self.remove_stopwords and word in self.stop_words and not word.endswith('_NEG'):
                continue
            pos = self.get_wordnet_pos(word)
            lemma = self.lemmatizer.lemmatize(word, pos)
            clean_tokens.append(lemma)
        return clean_tokens
    def _filter_short_words(self, tokens: List[str]) -> List[str]:
        return [w for w in tokens if len(w) > 2 or w.endswith('_NEG')]
    def _preprocess_transformer(self, text: str) -> str:
        text = self._remove_html_tags(text)
        text = self._remove_urls(text)
        text = self._normalize_whitespace(text)
        return text
    def _preprocess_classical(self, text: str) -> str:
        text = self._remove_html_tags(text)
        text = self._remove_urls(text)
        text = self._convert_emojis(text)
        text = self._expand_contractions(text)
        text = text.lower()
        text = self._replace_numbers(text)
        text = self._remove_non_alpha(text)
        tokens = self._tokenize(text)
        if self.use_negation:
            tokens = self._apply_negation(tokens)
        tokens = self._lemmatize_tokens(tokens)
        tokens = self._filter_short_words(tokens)
        return ' '.join(tokens)
    def clean_text(self, text: str) -> str:
        text = str(text)
        if self.mode == 'transformer':
            return self._preprocess_transformer(text)
        else:
            return self._preprocess_classical(text)
