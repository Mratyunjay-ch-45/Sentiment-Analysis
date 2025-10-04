
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs:", gpus)
except Exception as e:
    print("TensorFlow import error:", e)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from datasets import load_dataset
from transformers import BertTokenizer, TFBertForSequenceClassification

import nltk
nltk.download('punkt')


dataset = load_dataset('imdb')


import pandas as pd
train_df = pd.DataFrame({'text': dataset['train']['text'], 'label': dataset['train']['label']})
test_df  = pd.DataFrame({'text': dataset['test']['text'],  'label': dataset['test']['label']})

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df['label'].value_counts())


train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED, stratify=train_df['label'])
print("After split - Train:", train_df.shape, "Val:", val_df.shape)


print('\n----- SAMPLE POSITIVE -----')
print(train_df[train_df['label']==1]['text'].iloc[0][:500])
print('\n----- SAMPLE NEGATIVE -----')
print(train_df[train_df['label']==0]['text'].iloc[0][:500])

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(train_df['text'])
X_val_tfidf   = tfidf.transform(val_df['text'])
X_test_tfidf  = tfidf.transform(test_df['text'])


clf = LogisticRegression(max_iter=1000, random_state=SEED)
clf.fit(X_train_tfidf, train_df['label'])

def print_classification(y_true, y_pred):
    print(classification_report(y_true, y_pred, digits=4))

val_preds = clf.predict(X_val_tfidf)
print("\n--- TF-IDF + Logistic Regression (Validation) ---")
print_classification(val_df['label'], val_preds)

test_preds = clf.predict(X_test_tfidf)
print("\n--- TF-IDF + Logistic Regression (Test) ---")
print_classification(test_df['label'], test_preds)


import joblib
joblib.dump(clf, 'tfidf_logreg.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_VOCAB = 20000
MAX_LEN = 256
OOV_TOKEN = "<OOV>"

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(train_df['text'].tolist())

def texts_to_padded_sequences(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')

X_train_seq = texts_to_padded_sequences(train_df['text'])
X_val_seq   = texts_to_padded_sequences(val_df['text'])
X_test_seq  = texts_to_padded_sequences(test_df['text'])

y_train = train_df['label'].values
y_val   = val_df['label'].values
y_test  = test_df['label'].values

print('Shapes:', X_train_seq.shape, X_val_seq.shape, X_test_seq.shape)


import json
with open('keras_tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

EMBEDDING_DIM = 128

def build_rnn_model(rnn_type='LSTM', bidirectional=False, vocab_size=MAX_VOCAB, embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))

    rnn_layer = None
    if rnn_type.upper() == 'LSTM':
        rnn_layer = LSTM(128)
    elif rnn_type.upper() == 'GRU':
        rnn_layer = GRU(128)
    else:
        raise ValueError('Unknown rnn_type')

    if bidirectional:
        model.add(Bidirectional(rnn_layer))
    else:
        model.add(rnn_layer)

    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model_lstm = build_rnn_model('LSTM', bidirectional=False)
model_lstm.summary()


BATCH_SIZE = 64
EPOCHS = 6

callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]

history_lstm = model_lstm.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

lstm_test_preds = (model_lstm.predict(X_test_seq) > 0.5).astype(int).flatten()
print('\n--- LSTM Test Results ---')
print_classification(y_test, lstm_test_preds)


model_lstm.save('lstm_model.h5')


model_gru = build_rnn_model('GRU', bidirectional=False)
history_gru = model_gru.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

gru_test_preds = (model_gru.predict(X_test_seq) > 0.5).astype(int).flatten()
print('\n--- GRU Test Results ---')
print_classification(y_test, gru_test_preds)
model_gru.save('gru_model.h5')

model_bilstm = build_rnn_model('LSTM', bidirectional=True)
history_bilstm = model_bilstm.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

bilstm_test_preds = (model_bilstm.predict(X_test_seq) > 0.5).astype(int).flatten()
print('\n--- BiLSTM Test Results ---')
print_classification(y_test, bilstm_test_preds)
model_bilstm.save('bilstm_model.h5')


def plot_history(hist, title='Model'):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist.history['accuracy'], label='train_acc')
    plt.plot(hist.history['val_accuracy'], label='val_acc')
    plt.title(f'{title} Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'], label='train_loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.title(f'{title} Loss')
    plt.legend()
    plt.show()

plot_history(history_lstm, 'LSTM')
plot_history(history_gru, 'GRU')
plot_history(history_bilstm, 'BiLSTM')



TOKENIZER_NAME = 'bert-base-uncased'
MAX_LEN_BERT = 128

tokenizer_bert = BertTokenizer.from_pretrained(TOKENIZER_NAME)

def encode_texts(texts, tokenizer, max_len=MAX_LEN_BERT):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_len)
    return encodings

train_enc = encode_texts(train_df['text'], tokenizer_bert, MAX_LEN_BERT)
val_enc   = encode_texts(val_df['text'], tokenizer_bert, MAX_LEN_BERT)
test_enc  = encode_texts(test_df['text'], tokenizer_bert, MAX_LEN_BERT)


def to_tf_dataset(encodings, labels=None, batch_size=16):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(dict(encodings))
    else:
        ds = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    return ds.batch(batch_size)

train_ds = to_tf_dataset(train_enc, train_df['label'].values, batch_size=8)
val_ds   = to_tf_dataset(val_enc, val_df['label'].values, batch_size=8)
test_ds  = to_tf_dataset(test_enc, test_df['label'].values, batch_size=8)


bert_model = TFBertForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=2)


optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
bert_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


EPOCHS_BERT = 2
history_bert = bert_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_BERT)


bert_preds_logits = bert_model.predict(test_ds).logits
bert_preds = np.argmax(bert_preds_logits, axis=1)
print('\n--- BERT Test Results ---')
print_classification(test_df['label'].values, bert_preds)


bert_model.save_pretrained('tf_bert_sentiment')
tokenizer_bert.save_pretrained('tf_bert_tokenizer')


from sklearn.metrics import confusion_matrix

models_results = {
    'TFIDF_LogReg': test_preds,
    'LSTM': lstm_test_preds,
    'GRU': gru_test_preds,
    'BiLSTM': bilstm_test_preds,
    'BERT': bert_preds
}

results_summary = []
for name, preds in models_results.items():
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    results_summary.append({'model': name, 'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec})

results_df = pd.DataFrame(results_summary).sort_values('f1', ascending=False)
print('\nModel comparison:')
print(results_df)

# Plot confusion matrix for the best model (example: BERT)
cm = confusion_matrix(y_test, bert_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('BERT Confusion Matrix')
plt.show()



try:
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=['neg', 'pos'])

    # Baseline explain function
    def predict_proba_logreg(texts):
        X = tfidf.transform(texts)
        return clf.predict_proba(X)

    sample_idx = 5
    sample_text = test_df['text'].iloc[sample_idx]
    print('\nSample text (for LIME):', sample_text[:500])

    exp = explainer.explain_instance(sample_text, predict_proba_logreg, num_features=10)
    print(exp.as_list())
except Exception as e:
    print('LIME explanation skipped (missing package or error):', e)



