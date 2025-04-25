# Sentiment Analysis on Amazon Food Reviews

## Table of Contents
- [Project Overview](#project-overview)
- [Data Loading](#data-loading)
- [Data Cleaning](#data-cleaning)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Feature Importance](#feature-importance)
- [Model Persistence](#model-persistence)

## Project Overview
In this project, we develop a sentiment analysis pipeline to classify Amazon food reviews as **positive** or **negative**. We leverage Natural Language Processing (NLP) techniques to preprocess text, engineer features, and train classification models. Our goal is to identify the best-performing model for predicting review sentiment.

## Data Loading
1. **Dataset**: Amazon recommendation dataset (food reviews).
2. **Sentiment Labeling**:
   - Ignore all ratings equal to 3.  
   - Ratings > 3 → **positive** sentiment.  
   - Ratings < 3 → **negative** sentiment.
3. **Train/Test Split**: We partition the labeled data with an 80/20 train/test ratio using `train_test_split` from `scikit-learn`.

## Data Cleaning
We use Python’s `nltk` package to normalize text:
- **Stopwords Removal**: Eliminate common words (e.g., "the", "a").
- **Stemming**: Transform words to their stems (e.g., "tasty" → "tasti").
- **Punctuation Removal**: Strip punctuation (`.,?!`) while preserving emojis for sentiment cues.
- **Case Normalization**: Convert all text to lowercase.

## Feature Engineering
We convert cleaned text into numerical features:
1. **Count Vectorizer** (`CountVectorizer`): Unigrams and bigrams.
2. **TF-IDF Vectorizer** (`TfidfVectorizer`): Term frequency–inverse document frequency weighting.
3. **Word Embeddings**: (Optional) Word2Vec or similar embeddings.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
```

## Model Building
We explore classification algorithms using `scikit-learn`:
- **Logistic Regression**
- **Random Forest**
- (Optional) **Other classifiers**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
```

## Model Evaluation
We assess model performance using:
- **Accuracy**
- **Precision, Recall, F1-score**
- **ROC AUC**

```python
from sklearn import metrics
```  
Interpret metrics to balance false positives and false negatives.

## Feature Importance
After model selection, we inspect feature contributions:
- **Logistic Regression**: Use parameter coefficients to uncover the most influential tokens contributing to sentiment. 
- **Random Forest**: Use `feature_importances_` to identify key tokens driving sentiment.

## Model Persistence
To deploy without retraining:
1. **Serialize** the trained model using `joblib` or `pickle`.
2. **Load** the model for inference on new reviews.

```python
import joblib
joblib.dump(model, 'sentiment_model.pkl')
model = joblib.load('sentiment_model.pkl')
```
