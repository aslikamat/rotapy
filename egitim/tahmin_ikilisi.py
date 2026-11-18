import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

SEQUENCE_LEN = 6
FEATURE_COLS = [
    "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend",
    "lag_1h", "lag_24h", "lag_7d", "ma_3h", "current_speed",
    "is_event_time", "event_attendance", "event_radius_km", "hours_to_event",
    "temperature_c", "precipitation_mm", "wind_speed_kmh",
    "is_rainy", "is_snowy", "is_foggy", "is_bad_weather",
]

# Yükle
df        = pd.read_parquet("lstm_egitim_verisi.parquet")
model     = load_model("besiktas_lstm_model/besiktas_lstm.keras")
scaler_X  = pickle.load(open("besiktas_lstm_model/scaler_X.pkl", "rb"))
scaler_y  = pickle.load(open("besiktas_lstm_model/scaler_y.pkl", "rb"))

# Sequence oluştur
geohashler, pencereler = [], []
for gh, grp in df.groupby("geohash"):
    son6 = grp.tail(SEQUENCE_LEN)
    if len(son6) == SEQUENCE_LEN:
        geohashler.append(gh)
        pencereler.append(scaler_X.transform(son6[FEATURE_COLS].values))

# Tahmin
X      = np.stack(pencereler)
tahmin = scaler_y.inverse_transform(model.predict(X, verbose=0)).ravel().clip(0, 1)

# Sonuç — sadece geohash + congestion_ratio
sonuc = pd.DataFrame({
    "geohash":          geohashler,
    "congestion_ratio": tahmin.round(4),
})
sonuc = sonuc.sort_values("congestion_ratio", ascending=False).reset_index(drop=True)
sonuc.to_csv("tahmin_sonuc.csv", index=False)
print(sonuc.to_string(index=False))