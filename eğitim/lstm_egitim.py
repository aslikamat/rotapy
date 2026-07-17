"""
LSTM Model Eğitimi — Beşiktaş Trafik Yoğunluğu Tahmini
=========================================================
lstm_egitim_verisi.parquet dosyasını okur, LSTM modelini eğitir
ve eğitilmiş modeli kaydeder.

Kurulum:
    pip install pandas pyarrow numpy scikit-learn tensorflow matplotlib

Kullanım:
    python lstm_egitim.py                              # varsayılan
    python lstm_egitim.py --data lstm_egitim_verisi.parquet
    python lstm_egitim.py --epochs 20 --batch 64
    python lstm_egitim.py --test                       # 3 epoch hızlı test

Çıktı:
    besiktas_lstm_model/     → eğitilmiş model klasörü
    egitim_grafigi.png       → loss + tahmin grafiği
    model_sonuclari.json     → MAE, RMSE, doğruluk raporu

Birleştirme scriptindeki sütunlarla uyumlu:
    Anahtar : geohash, date_hour
    Hedef   : density
    Feature : hour_sin/cos, day_sin/cos, is_weekend,
              lag_1h, lag_24h, lag_7d, ma_3h, speed,
              is_event, event_attendance, event_radius_km, hours_to_event,
              temperature_c, precipitation_mm, wind_speed_kmh,
              cloud_cover_pct, humidity_pct,
              is_rainy, is_snowy, is_stormy, is_bad_weather
"""

import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =============================================================================
# AYARLAR
# =============================================================================

# birlestirme.py'deki LSTM_COLS ile örtüşen feature sütunları
FEATURE_COLS = [
    # Döngüsel zaman — hour/day yerine sin/cos kullanılıyor
    "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend",
    # Trafik geçmişi
    "lag_1h", "lag_24h", "lag_7d", "ma_3h", "speed",
    # Etkinlik — event_radius_km eklendi
    "is_event", "event_attendance", "event_radius_km", "hours_to_event",
    # Hava — cloud_cover_pct, humidity_pct, is_stormy eklendi
    "temperature_c", "precipitation_mm", "wind_speed_kmh",
    "cloud_cover_pct", "humidity_pct",
    "is_rainy", "is_snowy", "is_stormy", "is_bad_weather",
]

# Hedef değişken
TARGET_COL = "density"

# Gruplandırma anahtarı — farklı lokasyonların sequence'ları karışmasın
GROUP_COL = "geohash"   # birlestirme.py'de geohash var, location kaldırıldı

# Kaç önceki saate bakarak tahmin yapılsın
SEQUENCE_LEN = 6   # 24'ten 6'ya indirildi — rota önerisi için 30-60dk yeterli

# Train / Validation / Test oranları (zamana göre sıralı bölünür)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO = 0.15 (kalan)


# =============================================================================
# 1. VERİ YÜKLE ve HAZIRLA
# =============================================================================

def load_and_prepare(path: str) -> pd.DataFrame:
    print(f"\n[1/5] Veri yükleniyor: {path}")
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["datetime", GROUP_COL] if GROUP_COL in df.columns
                        else "datetime").reset_index(drop=True)
    print(f"  {len(df):,} satır, {len(df.columns)} sütun")

    # Eksik feature sütunları varsa 0 ile doldur
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    for col in missing:
        print(f"  [UYARI] '{col}' sütunu yok → 0 ile dolduruldu")
        df[col] = 0

    # Hedef sütun kontrolü
    if TARGET_COL not in df.columns:
        raise ValueError(f"Hedef sütun '{TARGET_COL}' bulunamadı!")

    print(f"  Hedef   : {TARGET_COL} — ort: {df[TARGET_COL].mean():.1f}, "
          f"max: {df[TARGET_COL].max():.0f}")
    if GROUP_COL in df.columns:
        print(f"  Geohash : {df[GROUP_COL].nunique()} benzersiz lokasyon")

    return df


# =============================================================================
# 2. SEQUENCE OLUŞTUR
# =============================================================================

def create_sequences(df: pd.DataFrame, scaler_X: MinMaxScaler,
                     scaler_y: MinMaxScaler, fit_scalers: bool = True) -> tuple:
    """
    LSTM için (X, y) sequence çiftleri oluşturur.

    Her X : son SEQUENCE_LEN saatin feature'ları  → shape: (SEQUENCE_LEN, n_features)
    Her y : bir sonraki saatin density değeri      → shape: (1,)

    Gruplandırma geohash üzerinden yapılır:
    farklı lokasyonların zaman serileri birbirine karışmaz.
    """
    print(f"\n[2/5] Sequence oluşturuluyor (pencere: {SEQUENCE_LEN} saat)...")

    X_all, y_all = [], []

    if GROUP_COL in df.columns:
        groups = df[GROUP_COL].unique()
    else:
        groups = ["_all"]

    for grp in groups:
        if GROUP_COL in df.columns:
            loc_df = df[df[GROUP_COL] == grp].copy()
        else:
            loc_df = df.copy()

        if len(loc_df) < SEQUENCE_LEN + 1:
            continue

        loc_df   = loc_df.sort_values("datetime")
        features = loc_df[FEATURE_COLS].values.astype(np.float32)
        targets  = loc_df[TARGET_COL].values.astype(np.float32)

        for i in range(SEQUENCE_LEN, len(loc_df)):
            X_all.append(features[i - SEQUENCE_LEN:i])
            y_all.append(targets[i])

    if not X_all:
        raise ValueError("Hiç sequence oluşturulamadı — veri çok az olabilir.")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32).reshape(-1, 1)

    print(f"  X shape : {X.shape}  (örnekler, zaman adımı, feature)")
    print(f"  y shape : {y.shape}")

    # Ölçeklendirme — 0-1 arasına sıkıştır
    X_2d = X.reshape(-1, X.shape[-1])
    if fit_scalers:
        X_scaled = scaler_X.fit_transform(X_2d)
        y_scaled = scaler_y.fit_transform(y)
    else:
        X_scaled = scaler_X.transform(X_2d)
        y_scaled = scaler_y.transform(y)

    X_scaled = X_scaled.reshape(X.shape)
    return X_scaled, y_scaled


# =============================================================================
# 3. TRAIN / VAL / TEST AYIR — ZAMANA GÖRE SIRALI
# =============================================================================

def split_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Veriyi zamana göre sıralı üçe böler — rastgele değil.

    Neden sıralı?
      Model geçmişten geleceği tahmin ediyor.
      Rastgele bölünürse model 2024 Aralık'tan öğrenip
      2023 Ocak'ı tahmin edebilir — gerçek dünyada işe yaramaz.

    Train  %70 → model bunlardan öğrenir
    Val    %15 → her epoch sonunda kontrol, EarlyStopping buraya bakar
    Test   %15 → model hiç görmedi, gerçek MAE buradan hesaplanır
    """
    n       = len(X)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    X_train, y_train = X[:n_train],                y[:n_train]
    X_val,   y_val   = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test,  y_test  = X[n_train + n_val:],        y[n_train + n_val:]

    print(f"\n[3/5] Veri bölündü (zamana göre sıralı):")
    print(f"  Train : {len(X_train):,} örnek  (%{TRAIN_RATIO*100:.0f})")
    print(f"  Val   : {len(X_val):,} örnek  (%{VAL_RATIO*100:.0f})")
    print(f"  Test  : {len(X_test):,} örnek  (%{(1-TRAIN_RATIO-VAL_RATIO)*100:.0f})")

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# 4. MODEL OLUŞTUR
# =============================================================================

def build_model(input_shape: tuple):
    """
    İki katmanlı LSTM + Dropout + Dense çıkış.
    Projenizin yöntem bölümüyle birebir örtüşüyor.

    input_shape = (SEQUENCE_LEN, n_features)
                = (6, 21)  — 6 saat geriye bak, 21 feature
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        # 1. LSTM katmanı — genel örüntüleri öğrenir
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),

        # 2. LSTM katmanı — daha ince örüntüleri öğrenir
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),

        # Normalizasyon — eğitimi stabilize eder
        BatchNormalization(),

        # Çıkış — tek sayı: tahmin edilen yoğunluk
        Dense(16, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    return model


# =============================================================================
# 5. EĞİT
# =============================================================================

def train(model, X_train, y_train, X_val, y_val,
          epochs: int = 30, batch_size: int = 32):
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )

    print(f"\n[4/5] Model eğitiliyor...")
    print(f"  Epochs    : {epochs}")
    print(f"  Batch     : {batch_size}")
    print(f"  Parametre : {model.count_params():,}")

    callbacks = [
        # Val loss 5 epoch iyileşmezse dur, en iyi ağırlıkları geri yükle
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        # Öğrenme hızını 3 epoch sonra yarıya indir
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        # Her epoch sonunda en iyi modeli diske yaz
        ModelCheckpoint(
            "en_iyi_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# =============================================================================
# 6. DEĞERLENDİR
# =============================================================================

def evaluate(model, X_test, y_test, scaler_y, history):
    print(f"\n[5/5] Model değerlendiriliyor (test seti)...")

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

    mean_val     = y_true.mean()
    hata_orani   = (mae / mean_val) * 100
    hedefe_ulasi = hata_orani < 20

    results = {
        "mae":           round(float(mae), 4),
        "rmse":          round(float(rmse), 4),
        "mape":          round(float(mape), 2),
        "hata_orani":    round(float(hata_orani), 2),
        "hedef_mae20":   hedefe_ulasi,
        "test_ornekler": len(y_test),
        "sequence_len":  SEQUENCE_LEN,
        "group_col":     GROUP_COL,
    }

    print(f"\n  {'='*45}")
    print(f"  SONUÇLAR")
    print(f"  {'='*45}")
    print(f"  MAE         : {mae:.4f} araç")
    print(f"  RMSE        : {rmse:.4f} araç")
    print(f"  MAPE        : %{mape:.2f}")
    print(f"  Hata oranı  : %{hata_orani:.2f}")
    print(f"  Hedef (%20) : {'✅ BAŞARILI' if hedefe_ulasi else '❌ Henüz değil'}")
    print(f"  {'='*45}")

    return results, y_pred, y_true


# =============================================================================
# KAYDET
# =============================================================================

def save_results(model, scaler_X, scaler_y, results, history,
                 y_pred, y_true, output_dir: str):
    import pickle
    import json

    Path(output_dir).mkdir(exist_ok=True)

    # Model
    model_path = f"{output_dir}/besiktas_lstm.keras"
    model.save(model_path)
    print(f"\n  Model kaydedildi      : {model_path}")

    # Scaler'lar — inference sırasında aynı ölçeklendirme gerekli
    with open(f"{output_dir}/scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(f"{output_dir}/scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)
    print(f"  Scaler'lar kaydedildi : {output_dir}/scaler_X.pkl, scaler_y.pkl")

    # Sonuç raporu
    results["feature_cols"] = FEATURE_COLS
    results["target_col"]   = TARGET_COL

    def to_serializable(obj):
        if isinstance(obj, (np.bool_, bool)):   return bool(obj)
        if isinstance(obj, np.integer):          return int(obj)
        if isinstance(obj, np.floating):         return float(obj)
        return obj

    results_clean = {k: to_serializable(v) for k, v in results.items()}
    with open(f"{output_dir}/model_sonuclari.json", "w", encoding="utf-8") as f:
        json.dump(results_clean, f, ensure_ascii=False, indent=2)
    print(f"  Rapor kaydedildi      : {output_dir}/model_sonuclari.json")

    # Eğitim grafiği
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history.history["loss"],     label="Train Loss")
        axes[0].plot(history.history["val_loss"], label="Val Loss")
        axes[0].set_title("Eğitim Loss (MSE)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE")
        axes[0].legend()
        axes[0].grid(True)

        n_show = min(500, len(y_true))
        axes[1].plot(y_true[:n_show], label="Gerçek",  alpha=0.7)
        axes[1].plot(y_pred[:n_show], label="Tahmin",  alpha=0.7)
        axes[1].set_title(f"Tahmin vs Gerçek (ilk {n_show} örnek)")
        axes[1].set_xlabel("Zaman adımı")
        axes[1].set_ylabel("Araç yoğunluğu (density)")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/egitim_grafigi.png", dpi=150)
        plt.close()
        print(f"  Grafik kaydedildi     : {output_dir}/egitim_grafigi.png")
    except Exception as e:
        print(f"  [UYARI] Grafik oluşturulamadı: {e}")

    print(f"\n  Tüm çıktılar: {output_dir}/")


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Beşiktaş LSTM Trafik Yoğunluğu Modeli"
    )
    parser.add_argument("--data",   default="lstm_egitim_verisi.parquet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--out",    default="besiktas_lstm_model")
    parser.add_argument("--test",   action="store_true",
                        help="3 epoch ile hızlı test")
    args = parser.parse_args()

    if args.test:
        args.epochs = 3
        print("[TEST MODU] 3 epoch ile hızlı test yapılıyor...")

    try:
        import tensorflow as tf
        print(f"\n  TensorFlow: {tf.__version__}")
    except ImportError:
        print("\n[HATA] TensorFlow kurulu değil!")
        print("pip install tensorflow")
        return

    # 1. Veri
    df = load_and_prepare(args.data)

    # 2. Scaler'lar
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # 3. Sequence
    X, y = create_sequences(df, scaler_X, scaler_y, fit_scalers=True)

    # 4. Böl
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # 5. Model
    model = build_model(input_shape=(SEQUENCE_LEN, len(FEATURE_COLS)))
    print(f"\n  Model özeti:")
    model.summary()

    # 6. Eğit
    history = train(
        model, X_train, y_train, X_val, y_val,
        epochs=args.epochs, batch_size=args.batch
    )

    # 7. Değerlendir
    results, y_pred, y_true = evaluate(model, X_test, y_test, scaler_y, history)

    # 8. Kaydet
    save_results(
        model, scaler_X, scaler_y,
        results, history, y_pred, y_true,
        output_dir=args.out
    )

    print(f"\n[TAMAMLANDI]")
    print(f"  Sonraki adım: modeli sistem/ klasörüne kopyalayın")
    print(f"  python ../sistem/tahmin.py --model {args.out}/besiktas_lstm.keras")


if __name__ == "__main__":
    main()
