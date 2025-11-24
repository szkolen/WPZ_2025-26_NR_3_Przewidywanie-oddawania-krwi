import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, classification_report
)
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Wczytanie i przygotowanie danych
# -------------------------
DATA_PATH = Path("transfusion.data")

column_names = [
    "Recency (months)",
    "Frequency (times)",
    "Monetary (c.c. blood)",
    "Time (months)",
    "target"
]

df = pd.read_csv(DATA_PATH, names=column_names, skiprows=1)

df = df.drop_duplicates().dropna().reset_index(drop=True)

# Nowe cechy
df["avg_blood_per_donation"] = df["Monetary (c.c. blood)"] / df["Frequency (times)"]
df["donations_per_month"] = df["Frequency (times)"] / df["Time (months)"]
df["recency_ratio"] = df["Recency (months)"] / df["Time (months)"]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

X = df.drop("target", axis=1)
y = df["target"].astype(int)

# Podział
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Rozmiary zbiorów -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# -------------------------
# Modele
# -------------------------
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(solver="liblinear", max_iter=1000))
])

pipe_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid_lr = {
    "clf__penalty": ["l1", "l2"],
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__class_weight": [None, "balanced"]
}

param_grid_rf = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 5],
    "clf__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_lr = GridSearchCV(pipe_lr, param_grid_lr, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1)
gs_rf = GridSearchCV(pipe_rf, param_grid_rf, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1)

print("Start GridSearch dla LogisticRegression...")
gs_lr.fit(X_train, y_train)
print("Done LR. Best params:", gs_lr.best_params_, "Best AUC:", gs_lr.best_score_)

print("Start GridSearch dla RandomForest...")
gs_rf.fit(X_train, y_train)
print("Done RF. Best params:", gs_rf.best_params_, "Best AUC:", gs_rf.best_score_)

# -------------------------
# Ocena modeli
# -------------------------
def evaluate_model(model, X, y, prefix=""):
    prob = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    results = {
        f"{prefix}auc": roc_auc_score(y, prob),
        f"{prefix}accuracy": accuracy_score(y, pred),
        f"{prefix}precision": precision_score(y, pred, zero_division=0),
        f"{prefix}recall": recall_score(y, pred, zero_division=0),
        f"{prefix}f1": f1_score(y, pred, zero_division=0),
        f"{prefix}mse": mean_squared_error(y, prob)
    }
    return results, pred, prob

best_models = {
    "LogisticRegression": gs_lr.best_estimator_,
    "RandomForest": gs_rf.best_estimator_
}

eval_rows = []
for name, model in best_models.items():
    val_res, val_pred, val_prob = evaluate_model(model, X_val, y_val, prefix="val_")
    test_res, test_pred, test_prob = evaluate_model(model, X_test, y_test, prefix="test_")

    row = {
        "model": name,
        **val_res,
        **test_res
    }
    eval_rows.append(row)

    print("\n=== Model:", name, "===\n")
    print("Validation metrics:")
    for k, v in val_res.items():
        print(f"  {k}: {v:.4f}")
    print("Test metrics:")
    for k, v in test_res.items():
        print(f"  {k}: {v:.4f}")
    print("Classification report (test):")
    print(classification_report(y_test, test_pred, zero_division=0))

results_df = pd.DataFrame(eval_rows)
results_df.to_csv("model_comparison_results.csv", index=False)
print("\nZapisano porównanie modeli do 'model_comparison_results.csv'")

# -------------------------
# Wybór NAJLEPSZEGO modelu → najniższy val_mse
# -------------------------
best_idx = results_df["val_mse"].idxmin()
best_row = results_df.loc[best_idx]
best_model_name = best_row["model"]
best_model = best_models[best_model_name]

joblib.dump(best_model, "best_model.joblib")
print(f"\nNajlepszy model (po najniższym MSE): {best_model_name}. Zapisano jako 'best_model.joblib'")

# -------------------------
# Raport
# -------------------------
summary = f"""
== Podsumowanie modelowania ==
Liczba rekordów (po czyszczeniu): {len(df)}

Modele porównane: LogisticRegression, RandomForest

Wyniki porównania:
{results_df.to_string(index=False)}

Wybrany model: {best_model_name}
Kryterium wyboru: najniższy błąd MSE na zbiorze walidacyjnym
"""
print(summary)

with open("modelling_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("Raport zapisano jako 'modelling_summary.txt'")
