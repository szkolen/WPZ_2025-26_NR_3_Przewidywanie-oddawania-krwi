import customtkinter as ctk
import pandas as pd
import joblib

# ---- Funkcja predykcji ----
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

# ---- Funkcja obsługi przycisku ----
def run_prediction():
    message_label.configure(text="", text_color="red")
    result_label.configure(text="")
    prob_label.configure(text="")

    try:
        recency = float(entry_recency.get())
        frequency = float(entry_frequency.get())
        monetary = float(entry_monetary.get())
        time = float(entry_time.get())

        # Walidacja logiczna
        if recency > time:
            message_label.configure(text="❌ Liczba miesięcy od ostatniej donacji nie może być większa\n niż liczba miesięcy od pierwszej donacji.")
            return
        if frequency <= 0 or time <= 0 or recency <= 0 or monetary < 0 :
            message_label.configure(text="❌ Wartości muszą być większe od 0.")
            return

        decision, prob = predict_donation(recency, frequency, monetary, time)
        result_label.configure(text=f"Czy osoba odda krew?: Prawdopodobnie {decision}", text_color="white")
        prob_label.configure(text=f"Prawdopodobieństwo, że osoba odda krew: {prob*100:.2f}%", text_color="white")

    except ValueError:
        message_label.configure(text="❌ Wprowadź prawidłowe liczby w każdym polu.")

# ---- UI ----
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Predykcja Oddania Krwi")
root.geometry("600x700")
root.resizable(False, False)

# Tytuł
title_label = ctk.CTkLabel(root, text="Predykcja oddania krwi w ciągu 6 miesięcy", font=("Arial", 22, "bold"))
title_label.pack(pady=20)

# Frame na pola
frame = ctk.CTkFrame(root, corner_radius=15)
frame.pack(pady=10, padx=30, fill="both", expand=False)

# Stała szerokość inputów
ENTRY_WIDTH = 200

# Pola wejściowe
labels_entries = [
    ("Liczba miesięcy od ostatniej donacji:", "recency"),
    ("Łączna liczba donacji:", "frequency"),
    ("Ilość oddanej krwi [ml]:", "monetary"),
    ("Liczba miesięcy od pierwszej donacji:", "time")
]

entries = {}
for text, key in labels_entries:
    lbl = ctk.CTkLabel(frame, text=text, anchor="w", font=("Arial", 14))
    lbl.pack(pady=(10, 2), padx=10)
    ent = ctk.CTkEntry(frame, width=ENTRY_WIDTH, height=35, font=("Arial", 14))
    ent.pack(pady=5, padx=10)
    entries[key] = ent

entry_recency = entries["recency"]
entry_frequency = entries["frequency"]
entry_monetary = entries["monetary"]
entry_time = entries["time"]

# Przycisk predykcji
predict_btn = ctk.CTkButton(root, text="Wykonaj predykcję", font=("Arial", 16, "bold"), height=45, command=run_prediction)
predict_btn.pack(pady=20, ipadx=10, ipady=5)

# Komunikaty 
message_label = ctk.CTkLabel(root, text="", font=("Arial", 14))
message_label.pack(pady=(5,0))

# Wyniki
result_label = ctk.CTkLabel(root, text="", font=("Arial", 18, "bold"))
result_label.pack(pady=(10,0))
prob_label = ctk.CTkLabel(root, text="", font=("Arial", 16))
prob_label.pack(pady=(5,20))

# Stopka
footer = ctk.CTkLabel(root, text="Model ML · Wydziałowy Projekt Zespołowy · WNIT AŁ", font=("Arial", 12))
footer.pack(pady=10)

root.mainloop()
