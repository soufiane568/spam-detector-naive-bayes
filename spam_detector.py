"""
Détecteur de Spam SMS avec Naïve Bayes
========================================
Projet de démonstration : classification binaire (spam / ham) sur des SMS,
utilisant l'algorithme Naïve Bayes multinomial, avec évaluation complète
(matrice de confusion, courbe ROC, AUC).

Dataset : SMS Spam Collection (5574 messages labellisés)
Auteur : Aimani Sofian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# -----------------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# -----------------------------------------------------------------------
print("=" * 60)
print("1. CHARGEMENT DES DONNÉES")
print("=" * 60)

df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "message"])
print(f"Nombre total de SMS : {len(df)}")
print(f"Répartition des classes :\n{df['label'].value_counts()}")
print(f"Proportion de spam : {(df['label'] == 'spam').mean():.2%}\n")

# Encodage binaire : spam = 1, ham = 0
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# -----------------------------------------------------------------------
# 2. SÉPARATION TRAIN / TEST
# -----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label_num"], test_size=0.25, random_state=42, stratify=df["label_num"]
)
print(f"Ensemble d'entraînement : {len(X_train)} messages")
print(f"Ensemble de test        : {len(X_test)} messages\n")

# -----------------------------------------------------------------------
# 3. VECTORISATION DU TEXTE (Bag of Words)
# -----------------------------------------------------------------------
print("=" * 60)
print("2. VECTORISATION (Bag of Words)")
print("=" * 60)

vectorizer = CountVectorizer(stop_words="english", lowercase=True, min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Taille du vocabulaire : {len(vectorizer.vocabulary_)} mots\n")

# -----------------------------------------------------------------------
# 4. ENTRAÎNEMENT DU MODÈLE NAÏVE BAYES
# -----------------------------------------------------------------------
print("=" * 60)
print("3. ENTRAÎNEMENT - Naïve Bayes Multinomial")
print("=" * 60)

model = MultinomialNB(alpha=1.0)  # alpha = lissage de Laplace
model.fit(X_train_vec, y_train)
print("Modèle entraîné.\n")

# -----------------------------------------------------------------------
# 5. PRÉDICTIONS
# -----------------------------------------------------------------------
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)[:, 1]  # probabilité classe "spam"

# -----------------------------------------------------------------------
# 6. ÉVALUATION - MÉTRIQUES
# -----------------------------------------------------------------------
print("=" * 60)
print("4. ÉVALUATION DU MODÈLE")
print("=" * 60)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy  : {acc:.4f}")
print(f"Précision : {prec:.4f}")
print(f"Rappel    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}\n")

print("Rapport de classification complet :")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# -----------------------------------------------------------------------
# 7. MATRICE DE CONFUSION
# -----------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(f"                Prédit ham   Prédit spam")
print(f"Réel ham        {cm[0][0]:<12} {cm[0][1]}")
print(f"Réel spam       {cm[1][0]:<12} {cm[1][1]}\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Matrice de Confusion")

# -----------------------------------------------------------------------
# 8. COURBE ROC
# -----------------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc:.3f})")
axes[1].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Aléatoire")
axes[1].set_xlabel("Taux de Faux Positifs")
axes[1].set_ylabel("Taux de Vrais Positifs")
axes[1].set_title("Courbe ROC")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("resultats_evaluation.png", dpi=150)
print("Graphiques sauvegardés dans 'resultats_evaluation.png'\n")

# -----------------------------------------------------------------------
# 9. TOP MOTS LES PLUS "SPAM" ET LES PLUS "HAM"
# -----------------------------------------------------------------------
print("=" * 60)
print("5. MOTS LES PLUS DISCRIMINANTS")
print("=" * 60)

feature_names = np.array(vectorizer.get_feature_names_out())
log_prob_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
top_spam_idx = np.argsort(log_prob_diff)[-15:][::-1]
top_ham_idx = np.argsort(log_prob_diff)[:15]

print("Top 15 mots associés au SPAM :")
print(", ".join(feature_names[top_spam_idx]))
print("\nTop 15 mots associés au HAM (messages normaux) :")
print(", ".join(feature_names[top_ham_idx]))

# -----------------------------------------------------------------------
# 10. TEST SUR DE NOUVEAUX MESSAGES
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. TEST SUR DE NOUVEAUX MESSAGES")
print("=" * 60)

nouveaux_messages = [
    "Congratulations! You've won a free iPhone. Click here to claim now!",
    "Hey, are we still meeting for coffee tomorrow at 5pm?",
    "URGENT: Your account will be suspended. Verify your details immediately by calling this number.",
    "Can you send me the notes from today's class?",
]

nouveaux_vec = vectorizer.transform(nouveaux_messages)
predictions = model.predict(nouveaux_vec)
probabilites = model.predict_proba(nouveaux_vec)[:, 1]

for msg, pred, proba in zip(nouveaux_messages, predictions, probabilites):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"[{label:4}] (p_spam={proba:.3f}) — {msg[:60]}...")
