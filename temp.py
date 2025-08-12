# region 1: Şifa Mah., Niyazi Mah. İsmetiye Mah., Dabakhane Mah., kavaklı bağ Mah.
# Hamidiye Mah., K. Mustafa paşa Mah., Ataköt Mah., Hacı Abdi Mah., Aslan Bey Mah., B. Mustafa Paşa Mah.,

# region 2: Yeni Hamam Mahj., Halfettin Mah., Akpınar Mah., Kırçuval Mah., Sancaktar Mah., 
# Büyük Hüseyinbey Mah., Küçük Hüseyinbey Mah., Saray Mah., Kernek Mah., İstiklal Mah., 
# İzzetiye Mah., Cevherizade Mah., Ferhadiye Mah., Hasan Varol Mah., Başharık Mah., Paşaköşkü Mah.

# region 3: Üçbağlar Mah., Zafer Mah., Fırat Mah., Yamaç Mah., Merkez Beydağı Mah., Selçuklu Mah.

# region 4: şehitfevzi Mah., beylerbaşı Mah., Sarıcıoğlu Mah., Cirikpınar Mah., taştepe Mah., 
# Hidayet Mah., iskender Mah.

# region 5: Battalgazi Mah., Göztepe Mah., Tandoğan Mah., Yıldıztepe Mah., çöşnük Mah.

# region 6: Orduzu Mah., Hanımınçiftliği Mah.

# region 7: Karaköy Mah., Bulgurlu Mah., Karakaş Çiftliği Mah., Göller Mah., Bağtepe Mah., 
# Hacı Yusuflar Mah., Karatepe Mah., Bağtepe Mah., Hacıhaliloğluçiftliği Mah., Yeniköy Mah., Çolaklı Mah.
# Selvidağ Mah., Yenice Mah., Tokluca Mah., Kapıkaya Mah., Bahri Mah., Merdivenler Mah., Düzyol Mah., kamıştaş Mah.
# Hisartepe Mah., Bulutlu Mah., Pelitli Mah., Alhanuşağı Mah.

# region 8: Tanışık Mah., Yaygın Mah., Gülümuşağı Mah., Uluköy Mah., Beydağı Mah., Fırıncı Mah., Üniversite Mah.,
# üzümlü Mah., Karagöz Mah., Çamurlu Mah.
 
import pandas as pd
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Veri seti (senin verdiğin)
# Mevcut data'nın devamı olarak ekle

data = {
    "m2_net": [130, 100, 215, 150, 220, 120, 85, 245, 125, 135, 145, 150, 160, 240, 190, 165, 60, 60, 500, 90,
               140, 110, 155, 170, 210, 125, 95, 260, 130, 145, 150, 155, 180, 230, 185, 175, 65, 70, 480, 100,
               135, 120, 165, 175, 200, 130, 100, 255, 135, 140, 150, 160, 190, 235, 195, 180, 68, 72, 470, 105,
               138, 115, 170, 180, 215, 140, 105, 250, 140, 150, 160, 165, 200, 240, 200, 185, 70, 75, 460, 110],

    "rooms": [4, 4, 5, 4, 5, 4, 3, 6, 3, 4, 4, 4, 4, 4, 4, 2, 2, 6, 3, 3,
              4, 3, 4, 5, 5, 4, 3, 6, 4, 4, 4, 4, 4, 5, 4, 4, 2, 3, 5, 3,
              4, 4, 4, 5, 5, 4, 3, 6, 4, 4, 4, 4, 4, 5, 4, 4, 2, 3, 5, 3,
              4, 4, 5, 5, 5, 4, 3, 6, 4, 4, 4, 4, 5, 5, 4, 4, 3, 2, 5, 3],

    "bathes": [1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1,
               1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1,
               1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1,
               1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1],

    "floor": [2, 3, 5, 4, 4, 1, 2, 1, 1, 2, 4, 9, 6, 2, 1, 7, 1, 3, 1, 1,
              3, 2, 5, 4, 4, 2, 1, 1, 3, 3, 5, 5, 2, 3, 1, 2, 1, 2, 4, 2,
              4, 3, 5, 4, 5, 2, 1, 1, 4, 3, 4, 5, 2, 3, 1, 2, 1, 2, 4, 3,
              3, 4, 5, 4, 5, 2, 1, 1, 3, 3, 5, 5, 2, 3, 1, 2, 1, 2, 4, 3],

    "elevator": [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
                 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
                 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
                 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],

    "aprtAge": [32, 26, 16, 20, 7, 6, 1, 31, 28, 16, 1, 6, 12, 14, 12, 6, 7, 12, 30, 1,
                15, 10, 14, 18, 8, 9, 5, 29, 13, 12, 11, 15, 16, 10, 7, 14, 5, 6, 20, 4,
                18, 11, 12, 15, 9, 10, 5, 28, 13, 12, 11, 14, 15, 10, 8, 7, 5, 6, 21, 4,
                17, 12, 13, 16, 10, 9, 4, 29, 14, 11, 10, 15, 16, 11, 9, 8, 6, 5, 20, 3],

    "region": [1, 2, 2, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6,
               1, 2, 2, 2, 3, 2, 3, 4, 4, 5, 5, 5, 5, 6, 5, 5, 6, 6, 7, 7,
               1, 2, 3, 2, 3, 2, 3, 4, 4, 5, 5, 5, 5, 6, 5, 5, 6, 6, 7, 7,
               1, 2, 3, 2, 3, 2, 3, 4, 4, 5, 5, 5, 5, 6, 5, 5, 6, 6, 7, 7],

    "heating": ["combi", "combi", "combi", "combi", "combi", "combi", "combi", "stove", "SE", "combi",
               "combi", "combi", "combi", "combi", "combi", "combi", "combi", "combi", "combi", "combi",
               "combi", "combi", "combi", "combi", "stove", "combi", "combi", "stove", "SE", "combi",
               "combi", "combi", "combi", "combi", "combi", "combi", "combi", "combi", "combi", "stove",
               "combi", "combi", "combi", "combi", "stove", "combi", "combi", "stove", "SE", "combi",
               "combi", "combi", "combi", "combi", "combi", "combi", "combi", "combi", "combi", "combi",
               "combi", "combi", "combi", "combi", "stove", "combi", "combi", "combi", "combi", "stove",
               "combi", "combi", "combi", "combi", "stove", "combi", "combi", "stove", "SE", "combi"],

    "balconies": [1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
                  1, 1, 1, 2, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2,
                  1, 1, 1, 2, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2,
                  1, 1, 1, 2, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2],

    "Price": [2700000, 1625000, 3500000, 2850000, 4175000, 2350000, 2650000, 4125000, 1688000,
              4750000, 3250000, 3200000, 2900000, 6000000, 3000000, 3700000, 1300000, 1000000,
              12000000, 1600000,
              2800000, 1700000, 3550000, 2950000, 4000000, 2450000, 2700000, 4200000, 1800000,
              4800000, 3300000, 3400000, 3000000, 6100000, 3100000, 3750000, 1350000, 1100000,
              11800000, 1650000,
              2850000, 1680000, 3600000, 3000000, 4050000, 2500000, 2750000, 4300000, 1850000,
              4850000, 3350000, 3450000, 3050000, 6150000, 3150000, 3800000, 1400000, 1150000,
              11900000, 1680000,
              2900000, 1750000, 3700000, 3100000, 4100000, 2550000, 2800000, 4350000, 1900000,
              4900000, 3400000, 3500000, 3100000, 6200000, 3200000, 3850000, 1450000, 1200000,
              12100000, 1700000]
}

for key, value in data.items():
    print(f"{key}: {len(value)}")



df = pd.DataFrame(data)
X = df.drop("Price", axis=1)
y = df["Price"]

categorical_features = ["heating"]
numeric_features = [col for col in X.columns if col not in categorical_features]
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# --- TKINTER ARAYÜZ ---
root = tk.Tk()
root.title("House Price Estimate")

# Fotoğraf ekleme
try:
    img = Image.open("house.jpg")  # kendi fotoğraf yolun
    img = img.resize((400, 300))
    photo = ImageTk.PhotoImage(img)
    tk.Label(root, image=photo).pack()
except:
    tk.Label(root, text="[The photo wasn't uploaded']").pack()

fields = {}
labels = ["Net m²", "The number of the rooms", "The number of the bathes", "The floor", "Elevator (0/1)", "The age of the apartment", "Region (1-8)", "Heating (combi/stove/SE)", "The number of the balconies"]
for label in labels:
    frame = tk.Frame(root)
    frame.pack(pady=2)
    tk.Label(frame, text=label, width=20, anchor="w").pack(side="left")
    entry = tk.Entry(frame)
    entry.pack(side="left")
    fields[label] = entry

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

def predict_price():
    try:
        m2_net = int(fields["Net m²"].get())
        rooms = int(fields["The number of the rooms"].get())
        bathes = int(fields["The number of the bathes"].get())
        floor = int(fields["The floor"].get())
        elevator = int(fields["Elevator (0/1)"].get())
        aprtAge = int(fields["The age of the apartment"].get())
        region = int(fields["Region (1-8)"].get())
        heating = fields["Heating (combi/stove/SE)"].get()
        balconies = int(fields["The number of the balconies"].get())

        new_house = pd.DataFrame({
            "m2_net": [m2_net],
            "rooms": [rooms],
            "bathes": [bathes],
            "floor": [floor],
            "elevator": [elevator],
            "aprtAge": [aprtAge],
            "region": [region],
            "heating": [heating],
            "balconies": [balconies]
        })

        prediction = model.predict(new_house)[0]
        result_label.config(text=f"Estimated Price: {prediction:,.0f} TL")
    except Exception as e:
        result_label.config(text=f"Hata: {e}")

tk.Button(root, text="Make a guess", command=predict_price, bg="green", fg="white").pack(pady=10)

root.mainloop()



