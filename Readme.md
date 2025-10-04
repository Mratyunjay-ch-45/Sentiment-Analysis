# IMDb Sentiment Analysis

This repository contains a comprehensive implementation of **sentiment analysis** on the IMDb movie reviews dataset using multiple NLP approaches, including **TF-IDF + Logistic Regression**, **RNNs (LSTM, GRU, BiLSTM)**, and **BERT**. It also includes visualization, model comparison, and interpretability with LIME.

---

## Table of Contents

- [Dataset](#dataset)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)


---

## Dataset

The project uses the [IMDb dataset](https://huggingface.co/datasets/imdb) available via the Hugging Face `datasets` library.  

- **Train set**: 25,000 movie reviews  
- **Test set**: 25,000 movie reviews  
- **Labels**: 0 (negative), 1 (positive)

---

## Features

- TF-IDF vectorization + Logistic Regression baseline  
- Tokenization and padding for RNN-based models  
- RNN architectures: LSTM, GRU, BiLSTM  
- Pretrained BERT (`bert-base-uncased`) fine-tuning  
- Model performance evaluation: Accuracy, Precision, Recall, F1-Score  
- Confusion matrix visualization  
- LIME-based interpretability for TF-IDF + Logistic Regression  
- Save and load models and tokenizers

---

## Requirements

Python 3.9+  

Key packages:

- `tensorflow`  
- `transformers`  
- `datasets`  
- `scikit-learn`  
- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `nltk`  
- `tqdm`  
- `joblib`  
- `lime` (optional, for interpretability)

Install dependencies via:

```bash
pip install -r requirements.txt

run the file using the command "python sentiment_analysis.py" from the code editor directly.
