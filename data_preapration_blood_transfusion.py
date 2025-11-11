# ====================================================
# ANALIZA I PRZYGOTOWANIE ZBIORU "Blood Transfusion Service Center"
# Plik źródłowy: transfusion.data
# ====================================================

import pandas as pd
from sklearn.model_selection import train_test_split

# === 1. Wczytanie danych ===
print(" Wczytywanie danych z pliku transfusion.data...")

# Nazwy kolumn wg dokumentacji UCI
column_names = [
    "Recency (months)",
    "Frequency (times)",
    "Monetary (c.c. blood)",
    "Time (months)",
    "whether he/she donated blood in March 2007"
]

df = pd.read_csv("transfusion.data", names=column_names, skiprows=1)
print("\n Dane wczytane poprawnie!")
print(df.head(), "\n")
print("Informacje o zbiorze:")
print(df.info(), "\n")

# === 2. Wstępna analiza danych ===
print("📊 Wstępna analiza danych:")
print(df.describe(), "\n")

print("Liczba braków danych:")
print(df.isna().sum(), "\n")

print("Rozkład zmiennej docelowej:")
print(df["whether he/she donated blood in March 2007"].value_counts(), "\n")

# === 3. Czyszczenie danych ===
print(" Czyszczenie danych...")

duplikaty = df.duplicated().sum()
print(f"Liczba duplikatów: {duplikaty}")

if duplikaty > 0:
    df = df.drop_duplicates()
    print(" Duplikaty usunięte.")
else:
    print(" Brak duplikatów.")

# Usuwanie braków danych, jeśli występują
if df.isnull().values.any():
    print(" Znaleziono braki danych — zostaną usunięte.")
    df = df.dropna()
else:
    print(" Brak braków danych.")

# === 4. Tworzenie nowych cech ===
print("\n🧮 Tworzenie nowych cech...")

df["avg_blood_per_donation"] = df["Monetary (c.c. blood)"] / df["Frequency (times)"]
df["donations_per_month"] = df["Frequency (times)"] / df["Time (months)"]
df["recency_ratio"] = df["Recency (months)"] / df["Time (months)"]

print(" Nowe cechy dodane:", list(df.columns), "\n")

# === 5. Podział na zbiory ===
print(" Podział na zbiory treningowy / walidacyjny / testowy...")

X = df.drop("whether he/she donated blood in March 2007", axis=1)
y = df["whether he/she donated blood in March 2007"]

# 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}\n")

# === 6. Raport z procesu ===
report = f"""
================ RAPORT PRZYGOTOWANIA DANYCH ================
Liczba rekordów po czyszczeniu: {len(df)}
Liczba duplikatów usuniętych: {duplikaty}
Braki danych: {df.isna().sum().sum()} (po czyszczeniu)
Nowe cechy: avg_blood_per_donation, donations_per_month, recency_ratio

Podział danych:
- Train: {X_train.shape[0]} rekordów
- Validation: {X_val.shape[0]} rekordów
- Test: {X_test.shape[0]} rekordów

Rozkład zmiennej docelowej (cały zbiór):
{y.value_counts(normalize=True).round(3)}

==============================================================
"""

print(report)

# Zapisanie raportu do pliku
with open("data_preparation_report.txt", "w") as f:
    f.write(report)

print(" Raport zapisany jako 'data_preparation_report.txt'")
print(" Dane gotowe do dalszej analizy lub modelowania.")
