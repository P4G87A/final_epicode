# DESCRIZIONE DEL PROGRAMMA
# Questo programma legge 50.000 recensioni di film del dataset IMDB
# (scaricato da Kaggle) e impara a capire se una recensione e'
# positiva o negativa.
#
# Come funziona:
# 1. Carica il dataset e pulisce il testo (toglie i tag HTML)
# 2. Divide i dati in training (80%) e test (20%)
# 3. Trasforma il testo in numeri con TF-IDF
# 4. Addestra due modelli: Naive Bayes e Logistic Regression
# 5. Confronta i modelli con accuracy, precision, recall e F1
# 6. Mostra la tabella  ConfusionMatrix
#
# Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# Il file "IMDB Dataset.csv" deve stare nella stessa cartella di questo script.

# ---------- LIBRERIE ----------
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


# ---------- 1. CARICAMENTO DEL DATASET ----------
# Cerco il file CSV nella stessa cartella dello script
cartella = os.path.dirname(os.path.abspath(__file__))
percorso_csv = os.path.join(cartella, "IMDB Dataset.csv")

# Se il file non c'e' avviso l'utente e fermo il programma
if not os.path.exists(percorso_csv):
    print("Errore: file 'IMDB Dataset.csv' non trovato in", cartella)
    print("Scaricalo da Kaggle e mettilo nella stessa cartella dello script.")
    exit()

df = pd.read_csv(percorso_csv)
print("Numero di recensioni:", len(df))
print(df["sentiment"].value_counts())  # quante positive e quante negative


# ---------- 2. PULIZIA DEL TESTO ----------
# Nelle recensioni ci sono dei tag HTML (es. <br />) che vanno tolti
def pulisci_testo(testo):
    testo = re.sub(r"<[^>]+>", " ", testo)  # rimuove i tag HTML
    testo = testo.lower()                   # tutto minuscolo
    return testo


df = df.drop_duplicates()                       # tolgo le recensioni duplicate
df["review"] = df["review"].apply(pulisci_testo)

# Trasformo le etichette in numeri: positive = 1, negative = 0
df["target"] = df["sentiment"].map({"positive": 1, "negative": 0})


# ---------- 3. DIVISIONE TRAINING / TEST ----------
X = df["review"]   # i testi
y = df["target"]   # le etichette (0 o 1)

# 80% per addestrare, 20% per testare
# stratify=y mantiene la stessa proporzione di positive e negative
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Recensioni di training:", len(X_train))
print("Recensioni di test:", len(X_test))


# ---------- 4. TRASFORMAZIONE DEL TESTO IN NUMERI (TF-IDF) ----------
# I modelli lavorano con i numeri, quindi trasformo ogni recensione in un vettore.
# TF-IDF da' piu' peso alle parole importanti e meno a quelle molto comuni.
vettorizzatore = TfidfVectorizer(
    stop_words="english",   # toglie parole comuni (the, and, is...)
    ngram_range=(1, 2),     # usa parole singole e coppie di parole
    min_df=2,               # ignora le parole che compaiono in meno di 2 recensioni
    max_features=100000     # massimo 100.000 caratteristiche
)

# fit_transform solo sul training, transform sul test
X_train_tfidf = vettorizzatore.fit_transform(X_train)
X_test_tfidf = vettorizzatore.transform(X_test)


# ---------- 5. ADDESTRAMENTO DEI MODELLI ----------
print("\nAddestro Naive Bayes...")
modello_nb = MultinomialNB()
modello_nb.fit(X_train_tfidf, y_train)

print("Addestro Logistic Regression...")
modello_lr = LogisticRegression(max_iter=1000)
modello_lr.fit(X_train_tfidf, y_train)


# ---------- 6. PREDIZIONI SUL TEST ----------
pred_nb = modello_nb.predict(X_test_tfidf)
pred_lr = modello_lr.predict(X_test_tfidf)


# ---------- 7. VALUTAZIONE ----------
print("\n=== NAIVE BAYES ===")
print(classification_report(y_test, pred_nb, target_names=["negative", "positive"]))

print("=== LOGISTIC REGRESSION ===")
print(classification_report(y_test, pred_lr, target_names=["negative", "positive"]))

# Tabella riassuntiva con le metriche dei due modelli
acc_nb = accuracy_score(y_test, pred_nb)
acc_lr = accuracy_score(y_test, pred_lr)

print("=== CONFRONTO MODELLI ===")
print("Naive Bayes         -> Accuracy:", round(acc_nb, 4),
      "| Precision:", round(precision_score(y_test, pred_nb), 4),
      "| Recall:", round(recall_score(y_test, pred_nb), 4),
      "| F1:", round(f1_score(y_test, pred_nb), 4))
print("Logistic Regression -> Accuracy:", round(acc_lr, 4),
      "| Precision:", round(precision_score(y_test, pred_lr), 4),
      "| Recall:", round(recall_score(y_test, pred_lr), 4),
      "| F1:", round(f1_score(y_test, pred_lr), 4))

# Scelgo il modello migliore in base all'accuracy
if acc_lr >= acc_nb:
    modello_migliore = modello_lr
    print("\nModello migliore: Logistic Regression")
else:
    modello_migliore = modello_nb
    print("\nModello migliore: Naive Bayes")


# ---------- 8. CONFUSION MATRIX ----------
# Mostra quante recensioni sono state classificate bene o male
fig, assi = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay.from_predictions(
    y_test, pred_nb, display_labels=["negative", "positive"],
    cmap="Blues", ax=assi[0], colorbar=False
)
assi[0].set_title("Naive Bayes")

ConfusionMatrixDisplay.from_predictions(
    y_test, pred_lr, display_labels=["negative", "positive"],
    cmap="Blues", ax=assi[1], colorbar=False
)
assi[1].set_title("Logistic Regression")

plt.tight_layout()


# Mostro il grafico a schermo (chiudi la finestra per terminare il programma)
plt.show()