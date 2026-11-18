import csv

from collections import Counter

from pathlib import Path

import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report


PROJECT_DIR = Path(__file__).parent

DATA_PATH = PROJECT_DIR / "data" / "gestures.csv"

MODEL_PATH = PROJECT_DIR / "models" / "gesture_classifier.joblib"

with DATA_PATH.open("r", newline="", encoding="utf-8") as file:

    rows = list(csv.DictReader(file))

if not rows:

    raise ValueError("Le fichier gestures.csv ne contient aucune donnée.")

labels = [row["label"] for row in rows]

counts = Counter(labels)

print("Exemples par geste :", dict(counts))

features = [

    [float(row[column]) for column in rows[0] if column != "label"]

    for row in rows

]

if min(counts.values()) < 2:

    raise ValueError("Il faut au moins 2 exemples par geste pour entraîner le modèle.")

X_train, X_test, y_train, y_test = train_test_split(

    features,

    labels,

    test_size=0.2,

    random_state=42,

    stratify=labels,

)

model = RandomForestClassifier(

    n_estimators=200,

    class_weight="balanced",

    random_state=42,

)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"\nPrécision sur les données de test : {accuracy_score(y_test, predictions):.2%}")

print(classification_report(y_test, predictions, zero_division=0))

# On réentraîne avec toutes les données avant de sauvegarder le modèle.

model.fit(features, labels)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print(f"Modèle enregistré dans : {MODEL_PATH}")
 