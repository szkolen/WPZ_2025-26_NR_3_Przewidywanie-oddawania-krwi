import pandas as pd
import joblib

# -------------------------
# Funkcja do predykcji
# -------------------------
def predict_donation(recency, frequency, monetary, time):
    model = joblib.load("best_model.joblib")

    data = pd.DataFrame([{
        "Recency (months)": recency,
        "Frequency (times)": frequency,
        "Monetary (c.c. blood)": monetary,
        "Time (months)": time,
        "avg_blood_per_donation": monetary / frequency if frequency else 0,
        "donations_per_month": frequency / time if time else 0,
        "recency_ratio": recency / time if time else 0
    }])

    prob = model.predict_proba(data)[0, 1]
    decision = "TAK" if prob >= 0.5 else "NIE"

    return decision, float(prob)


# -------------------------
# Program główny – pytania do użytkownika
# -------------------------
print("=== Program predykcji oddania krwi w ciągu 6 miesięcy ===\n")

recency = float(input("1. Ile miesięcy temu oddano ostatnio krew? "))
frequency = float(input("2. Ile razy osoba oddała krew w życiu? "))
monetary = float(input("3. Ile łącznie oddanej krwi (w c.c.)? "))
time = float(input("4. Ile miesięcy minęło od pierwszej donacji? "))

decision, prob = predict_donation(recency, frequency, monetary, time)

print("\n=== WYNIK ===")
print(f"Czy osoba odda krew w ciągu 6 miesięcy? → {decision}")
print(f"Prawdopodobieństwo oddania krwi: {prob * 100:.2f}%")
