# IMDb Sentiment Analysis

This repository contains a comprehensive implementation of **sentiment analysis** on the IMDb movie reviews dataset using multiple NLP approaches, including **TF-IDF + Logistic Regression**, **Recurrent Neural Networks (RNNs: LSTM, GRU, BiLSTM)**, and **BERT**. It also includes visualization, model comparison, and interpretability with LIME.

---

## Table of Contents

- [Dataset](#dataset)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)

---

## Dataset

The project uses the [IMDb dataset](https://huggingface.co/datasets/imdb) available via the Hugging Face `datasets` library. 

* **Train set**: 25,000 movie reviews 
* **Test set**: 25,000 movie reviews 
* **Labels**: 0 (negative), 1 (positive)

---

## Features

* **Baseline Model**: TF-IDF vectorization + Logistic Regression.
* **RNN Models**: Tokenization, padding, and training with **LSTM**, **GRU**, and **BiLSTM** architectures.
* **Transformer Model**: Fine-tuning of the pretrained **BERT** (`bert-base-uncased`) model.
* **Evaluation**: Model performance evaluation using **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
* **Visualization**: Confusion matrix visualization.
* **Interpretability**: LIME-based interpretability for the TF-IDF + Logistic Regression model.
* **Utility**: Save and load trained models and tokenizers.

---

## Requirements

The project requires **Python 3.9+**.

Key packages:

* `tensorflow`
* `transformers`
* `datasets`
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `nltk`
* `tqdm`
* `joblib`
* `lime` (optional, for interpretability)

---

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/yourusername/imdb-sentiment-analysis.git](https://github.com/yourusername/imdb-sentiment-analysis.git)
    cd imdb-sentiment-analysis
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

To run the main sentiment analysis script and train the models:

```bash
python sentiment_analysis.py