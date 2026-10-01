"""
lstm_inceleme.py — lstm_egitim.py'deki modelin içinde ne olduğunu adım adım gösterir
=====================================================================================

Bu script MODEL EĞİTMEZ. Amacı şu soruları, bizim kodumuzdan ve kullandığımız
kütüphanenin (Keras) kendi kaynak kodundan göstererek cevaplamak:

    BÖLÜM 1 → 150 saatlik veri, bizim create_sequences() ile nasıl pencerelere bölünüyor?
    BÖLÜM 2 → build_model()'deki LSTM(units=64) satırı Keras'ın içinde neye dönüşüyor?
    BÖLÜM 3 → units=64 neden "64 hücre" değil, ağırlık matrislerinin boyutu?
    BÖLÜM 4 → h_0 ve c_0 neden sıfır? (Keras'ın hangi satırı bunu yapıyor?)
    BÖLÜM 5 → Tek hücrenin içinde f, i, g, o kapıları nasıl hesaplanıyor?
              (Keras'ın kodunu numpy ile birebir tekrar yazıp sonucu karşılaştırıyoruz)
    BÖLÜM 6 → 6 zaman adımı boyunca "aynı hücrenin tekrar çalışması" nerede oluyor?
    BÖLÜM 7 → return_sequences: 1. katmandan 6, 2. katmandan 1 vektör neden çıkıyor?
    BÖLÜM 8 → Dropout, BatchNormalization ve Dense katmanları ne yapıyor?
    BÖLÜM 9 → batch_size=32 eğitim sırasında nasıl uygulanıyor?


tensorflow / keras
    Sinir ağı kütüphanesi. Keras, TensorFlow'un içindeki "kolay kullanım" kısmı.
    Bir katman (LSTM, Dense, Dropout...) aslında bir Python SINIFIDIR (class):
        __init__()  → katman oluşturulurken çalışır, ayarları kaydeder (units vb.)
        build()     → girdinin boyutu ilk kez görüldüğünde çalışır, AĞIRLIKLARI oluşturur
        call()      → her veri geçtiğinde çalışır, HESABI yapar
    Keras içinde "ops.matmul", "ops.zeros" gibi fonksiyonlar görürsünüz; bunlar
    numpy'daki "@" ve "np.zeros" ile aynı işi yapar, sadece TensorFlow tensörleriyle.

inspect (Python'un standart kütüphanesi)
    Bir fonksiyonun KAYNAK KODUNU okumamızı sağlar. Bu scriptte, Keras'ın
    bilgisayarınızda kurulu olan sürümünün gerçek kodunu satır numaralarıyla
    ekrana basmak için kullanıyoruz. Böylece "kütüphane bunu böyle yapıyor"
    dediğimiz her şeyi kendi gözünüzle görebilirsiniz.
"""

import os
import inspect

# TensorFlow açılırken çok sayıda teknik log basar; sadece hataları göster.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Keras'ın LSTM'i aslında iki sınıftan oluşur:
#   LSTMCell → TEK bir zaman adımının hesabını yapan hücre (kapılar burada)
#   RNN      → hücreyi zaman boyunca döngüde çağıran "sarmalayıcı"
#   LSTM     → RNN'in, içine LSTMCell konmuş hali (bizim kodda kullandığımız)
from keras.src.layers.rnn.lstm import LSTM, LSTMCell
from keras.src.layers.rnn.rnn import RNN
from keras.src.layers.core.dense import Dense
from keras.src.layers.regularization.dropout import Dropout

# BİZİM KODUMUZ: lstm_egitim.py'den doğrudan fonksiyon ve ayarları alıyoruz.
# Böylece burada incelediğimiz model, eğitimde kullanılanla birebir aynı.
from lstm_egitim import (
    build_model,        # modeli kuran fonksiyon
    create_sequences,   # veriyi 6 saatlik pencerelere bölen fonksiyon
    FEATURE_COLS,       # 23 özellik sütunu
    TARGET_COL,         # "congestion_ratio"
    GROUP_COL,          # "geohash"
    SEQUENCE_LEN,       # 6
    TRAIN_RATIO,        # 0.70
)

np.set_printoptions(precision=4, suppress=True, linewidth=100)
np.random.seed(0)
tf.random.set_seed(0)

N_FEAT = len(FEATURE_COLS)   # 23

import keras
print(f"Keras sürümü: {keras.__version__}  (gösterilen kaynak kod satır numaraları bu sürüme aittir)")


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def baslik(metin):
    print("\n\n" + "=" * 80)
    print(metin)
    print("=" * 80)


def keras_kodu_goster(fonksiyon, baslangic=None, satir_sayisi=None, bitis=None):
    """
    Keras'ın kurulu sürümündeki bir fonksiyonun kaynak kodunu, dosyadaki gerçek
    satır numaralarıyla ekrana basar.

    fonksiyon    : örn. LSTMCell.get_initial_state
    baslangic    : bu metni içeren satırdan itibaren göster (None → fonksiyonun başı)
    satir_sayisi : kaç satır gösterilsin
    bitis        : bu metni içeren satıra kadar göster

    inspect.getsourcelines(f) → (satır listesi, ilk satırın numarası) döndürür.
    inspect.getsourcefile(f)  → fonksiyonun bulunduğu .py dosyasının yolu.
    """
    satirlar, ilk_no = inspect.getsourcelines(fonksiyon)
    dosya = inspect.getsourcefile(fonksiyon)
    # Dosya yolunu "keras/src/..." kısmından itibaren kısaltalım
    kisa = dosya[dosya.find("keras"):] if "keras" in dosya else dosya

    bas = 0
    if baslangic is not None:
        bulunan = [k for k, s in enumerate(satirlar) if baslangic in s]
        if not bulunan:
            print(f"  [Not] '{baslangic}' bu Keras sürümünde bulunamadı, tüm fonksiyon gösteriliyor.")
        else:
            bas = bulunan[0]
    son = len(satirlar)
    if bitis is not None:
        bulunan = [k for k, s in enumerate(satirlar[bas:]) if bitis in s]
        if bulunan:
            son = bas + bulunan[0] + 1
    elif satir_sayisi is not None:
        son = min(len(satirlar), bas + satir_sayisi)

    print(f"  ┌─ Keras kaynak kodu: {kisa}")
    for k in range(bas, son):
        print(f"  │ {ilk_no + k:>4} │ {satirlar[k].rstrip()}")
    print("  └" + "─" * 60)


# =============================================================================
# BÖLÜM 1 — VERİ: 150 saat → 6 saatlik pencereler
# =============================================================================
baslik("BÖLÜM 1 — 150 saatlik veri, bizim create_sequences() ile pencerelere bölünüyor")

# Gerçek eğitim verisi (lstm_egitim_verisi.parquet) repoda olmadığı için, onunla
# AYNI SÜTUNLARA sahip sahte bir tablo oluşturuyoruz: 1 geohash, 150 saat.
# pandas.DataFrame = Excel tablosu gibi satır/sütunlu veri yapısı.
saat_sayisi = 150
df = pd.DataFrame(
    np.random.rand(saat_sayisi, N_FEAT),   # 150 satır × 23 sütun rastgele sayı
    columns=FEATURE_COLS,                  # sütun isimleri bizim FEATURE_COLS
)
df["datetime"] = pd.date_range("2024-01-01", periods=saat_sayisi, freq="h")
df[GROUP_COL] = "sxk9s3"                   # README'deki geohash'lerden biri
df[TARGET_COL] = np.random.rand(saat_sayisi)

print(f"Sahte veri tablosu: {df.shape[0]} satır (saat), "
      f"{len(FEATURE_COLS)} özellik + datetime + geohash + {TARGET_COL}")

# >>> BİZİM KODUMUZ ÇALIŞIYOR: lstm_egitim.py → create_sequences()
# İçindeki döngü (lstm_egitim.py, satır ~157-159):
#     for i in range(SEQUENCE_LEN, len(loc_df)):
#         X_all.append(features[i - SEQUENCE_LEN:i])   # i'den önceki 6 saat
#         y_all.append(targets[i])                     # i. saatin hedefi
# i = 6, 7, ..., 149 → 150 - 6 = 144 pencere
X, y = create_sequences(df, MinMaxScaler(), MinMaxScaler(), fit_scalers=True)

print(f"\nSonuç: X.shape = {X.shape}  → (pencere sayısı, saat, özellik)")
print(f"       y.shape = {y.shape}   → her pencere için 1 hedef değer")
print("\nPencereler 1'er saat kayarak oluşuyor (üst üste biniyor):")
for k in [0, 1, 2, len(X) - 1]:
    print(f"  Pencere {k+1:>3}: saat {k+1:>3}–{k+6:<3} → hedef: saat {k+7}")


# =============================================================================
# BÖLÜM 2 — build_model() içindeki LSTM(units=64) satırı Keras'ta neye dönüşüyor?
# =============================================================================
baslik("BÖLÜM 2 — LSTM(units=64, ...) yazınca Keras'ın içinde ne oluşuyor?")

# >>> BİZİM KODUMUZ: lstm_egitim.py → build_model()
model = build_model(input_shape=(SEQUENCE_LEN, N_FEAT))
lstm1 = model.layers[0]      # LSTM(units=64, return_sequences=True)
dropout1 = model.layers[1]   # Dropout(0.2)
lstm2 = model.layers[2]      # LSTM(units=32, return_sequences=False)

print("""
Bizim kodda tek satır yazıyoruz:
    LSTM(units=64, return_sequences=True, ...)

Keras'ın LSTM sınıfı bu satırı iki parçaya ayırır:
  1) Bir LSTMCell oluşturur ve units=64'ü ONA verir   → hesabı yapan TEK hücre
  2) Bu hücreyi RNN sınıfına verir                     → hücreyi 6 kez döngüde çağıran yapı
LSTM.__init__ içindeki ilgili kısım:""")
keras_kodu_goster(LSTM.__init__, baslangic="cell = LSTMCell(", satir_sayisi=4)
keras_kodu_goster(LSTM.__init__, baslangic="super().__init__(", satir_sayisi=4)

print(f"""
Kanıt — bizim modelimizdeki 1. katmanın içinde kaç hücre var?
  type(lstm1)          = {type(lstm1).__name__}
  type(lstm1.cell)     = {type(lstm1.cell).__name__}     ← tek bir hücre nesnesi (liste değil)
  lstm1.cell.units     = {lstm1.cell.units}
  lstm2.cell.units     = {lstm2.cell.units}""")


# =============================================================================
# BÖLÜM 3 — units=64 ne işe yarıyor? (Ağırlık matrisleri)
# =============================================================================
baslik("BÖLÜM 3 — units=64 hücre sayısı değil: ağırlık matrislerinin boyutunu belirliyor")

print("""
LSTMCell.__init__ içinde units şu şekilde kaydediliyor. state_size = [64, 64]
"hücrenin 2 hafıza vektörü var, ikisi de 64 uzunluğunda" demek (h ve c):""")
keras_kodu_goster(LSTMCell.__init__, baslangic="self.units = units", satir_sayisi=1)
keras_kodu_goster(LSTMCell.__init__, baslangic="self.state_size", satir_sayisi=2)

print("""
Ağırlıklar build() içinde oluşturuluyor. input_dim = girdinin son boyutu
(1. katman için 23 özellik). "self.units * 4" → 4 kapı (i, f, g, o) için yan yana
4 tane 64'lük blok:""")
keras_kodu_goster(LSTMCell.build, baslangic="input_dim = input_shape[-1]",
                  bitis='name="recurrent_kernel"')

for ad, katman, girdi_ad in [("LSTM-1", lstm1, "23 özellik"),
                             ("LSTM-2", lstm2, "LSTM-1'in 64'lük h'si")]:
    W = katman.cell.kernel.numpy()            # x_t ile çarpılan matris
    U = katman.cell.recurrent_kernel.numpy()  # h_(t-1) ile çarpılan matris
    b = katman.cell.bias.numpy()
    u = katman.cell.units
    print(f"\n{ad} (units={u}, girdi = {girdi_ad}):")
    print(f"  kernel           W.shape = {W.shape}   = (input_dim, units*4)")
    print(f"  recurrent_kernel U.shape = {U.shape}   = (units, units*4)")
    print(f"  bias             b.shape = {b.shape}      = (units*4,)")
    print(f"  Sütunların kapılara dağılımı: "
          f"i=[0:{u}]  f=[{u}:{2*u}]  g=[{2*u}:{3*u}]  o=[{3*u}:{4*u}]")
    elle = W.size + U.size + b.size
    print(f"  Parametre: {W.size} + {U.size} + {b.size} = {elle:,}  "
          f"(Keras: {katman.count_params():,})")

print(f"""
Parametre sayısında SEQUENCE_LEN (6) hiç geçmiyor. Yani 6 zaman adımı için ayrı
ağırlık yok; 6 adımın hepsi bu TEK W, U, b üçlüsünü kullanıyor.""")

print("""
Ek bilgi — bias başlangıç değeri. LSTMCell'de unit_forget_bias=True varsayılan;
forget kapısının bias'ı 0 değil 1 ile başlatılıyor (diğerleri 0):""")
keras_kodu_goster(LSTMCell.build, baslangic="if self.unit_forget_bias:",
                  bitis="else:")
b1 = lstm1.cell.bias.numpy()
print(f"  Bizim modelde bias'ın ilk değerleri:")
print(f"    i bloğu  b[0:4]     = {b1[0:4]}")
print(f"    f bloğu  b[64:68]   = {b1[64:68]}   ← 1'ler")
print(f"    g bloğu  b[128:132] = {b1[128:132]}")
print(f"    o bloğu  b[192:196] = {b1[192:196]}")
print("  Amaç: eğitimin başında hücre 'hafızayı unutmaya' değil 'tutmaya' meyilli olsun.")


# =============================================================================
# BÖLÜM 4 — h_0 ve c_0 neden sıfır?
# =============================================================================
baslik("BÖLÜM 4 — h_0 = c_0 = 0: Keras'ın hangi satırı bunu yapıyor?")

print("""
Bizim kodda LSTM katmanını çağırırken başlangıç hafızası (initial_state) VERMİYORUZ.
Keras'ın RNN.call() fonksiyonu bu durumda şu kontrolü yapıyor:""")
keras_kodu_goster(RNN.call, baslangic="if initial_state is None:", satir_sayisi=7)

print(f"""
Bizim modelde:  lstm1.stateful = {lstm1.stateful}
→ "else" dalına girilir → self.get_initial_state(...) çağrılır.
batch_size = ops.shape(sequences)[0] → o an işlenen pencere sayısı (örn. 32).

RNN.get_initial_state, hücrenin kendi get_initial_state'ini çağırır:""")
keras_kodu_goster(LSTMCell.get_initial_state)

print(f"""
self.state_size = [64, 64] olduğundan bu kod şunu üretiyor:
    [ ops.zeros((batch_size, 64)),    ← h_0
      ops.zeros((batch_size, 64)) ]   ← c_0
Yani her pencere için 64 sıfırlık h_0 ve 64 sıfırlık c_0.

Bu çağrı RNN.call()'un her çalışmasında, yani HER BATCH'TE yeniden yapılır.
Bu yüzden bir pencerenin hafızası bir sonrakine taşınmaz.

Kanıt — Keras'a aynı fonksiyonu bizim modelimiz üzerinden çağırtalım:""")
baslangic_durumu = lstm1.get_initial_state(batch_size=2)
for ad, d in zip(["h_0", "c_0"], baslangic_durumu):
    d = np.array(d)
    print(f"  {ad}.shape = {d.shape},  tüm değerler sıfır mı? {np.all(d == 0)}")


# =============================================================================
# BÖLÜM 5 — Tek hücrenin içi: f, i, g, o kapıları
# =============================================================================
baslik("BÖLÜM 5 — Hücrenin içi: Keras'ın call() kodu ve numpy ile birebir tekrarı")

print("""
Bir zaman adımında hücrenin yaptığı hesap LSTMCell.call() içinde. Bizim modelde
varsayılan implementation=2 kullanıldığı için şu kısım çalışıyor:""")
keras_kodu_goster(LSTMCell.call, baslangic="z = ops.matmul(inputs, self.kernel)",
                  bitis="return h, [h, c]")
print("\nz'nin 4 parçaya bölündükten sonra işlendiği yer:")
keras_kodu_goster(LSTMCell._compute_carry_and_output_fused)

print("""
Satır satır anlamı:
  z = x_t @ W            → 23'lük girdi × (23, 256)    = 256 sayı
  z = z + h_(t-1) @ U    → 64'lük hafıza × (64, 256)   = 256 sayı, toplanır
  z = z + b              → 256'lık bias eklenir
  z0,z1,z2,z3 = split(z) → 256 sayı 4 eşit parçaya: her biri 64
  i = sigmoid(z0)        → input gate:  0..1, "yeni bilgiden ne kadar yazayım?"
  f = sigmoid(z1)        → forget gate: 0..1, "eski hafızadan ne kadar tutayım?"
  g = tanh(z2)           → aday değer: -1..1, "yazılacak yeni bilgi"
  o = sigmoid(z3)        → output gate: 0..1, "hafızadan ne kadar dışarı vereyim?"
  c = f*c_(t-1) + i*g    → yeni uzun vadeli hafıza (64)
  h = o * tanh(c)        → yeni çıktı / kısa vadeli hafıza (64)

recurrent_activation=sigmoid ve activation=tanh — bizim build_model()'de
açıkça yazdığımız değerler; Keras'ta zaten varsayılan:""")
keras_kodu_goster(LSTMCell.__init__, baslangic='activation="tanh"', satir_sayisi=2)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def hucre_bir_adim(x_t, h_onceki, c_onceki, W, U, b, units):
    """Keras'taki LSTMCell.call + _compute_carry_and_output_fused'un numpy hali."""
    z = x_t @ W + h_onceki @ U + b            # (256,)
    z0 = z[0 * units:1 * units]
    z1 = z[1 * units:2 * units]
    z2 = z[2 * units:3 * units]
    z3 = z[3 * units:4 * units]
    i = sigmoid(z0)
    f = sigmoid(z1)
    g = np.tanh(z2)
    o = sigmoid(z3)
    c = f * c_onceki + i * g
    h = o * np.tanh(c)
    return h, c, (i, f, g, o)


# Bölüm 1'deki GERÇEK ilk penceremizi (saat 1-6) alalım
pencere = X[0]                                 # (6, 23)
W = lstm1.cell.kernel.numpy()
U = lstm1.cell.recurrent_kernel.numpy()
b = lstm1.cell.bias.numpy()

h = np.zeros(64, dtype=np.float32)             # Bölüm 4: h_0 = 0
c = np.zeros(64, dtype=np.float32)             # Bölüm 4: c_0 = 0
x1 = pencere[0]                                # saat 1'in 23 özelliği

print(f"\nÖrnek: Pencere 1'in ilk saati (x_1) ile tek adım:")
print(f"  x_1.shape = {x1.shape},  h_0.shape = {h.shape},  c_0.shape = {c.shape}")
h1, c1, (i, f, g, o) = hucre_bir_adim(x1, h, c, W, U, b, 64)
print(f"  z = x_1 @ W + h_0 @ U + b  → shape {(x1 @ W).shape}")
print(f"  i[:4] = {i[:4]}")
print(f"  f[:4] = {f[:4]}   ← 64 değerin ortalaması {f.mean():.2f}; bias 1 ile başladığı için")
print(f"                                      sigmoid(1)≈0.73 civarında toplanıyor")
print(f"  g[:4] = {g[:4]}")
print(f"  o[:4] = {o[:4]}")
print(f"  c_1 = f*c_0 + i*g → c_0 sıfır olduğu için c_1 = i*g  → c_1[:4] = {c1[:4]}")
print(f"  h_1 = o*tanh(c_1)                                  → h_1[:4] = {h1[:4]}")


# =============================================================================
# BÖLÜM 6 — "Aynı hücre 6 kez" döngüsü nerede?
# =============================================================================
baslik("BÖLÜM 6 — Aynı hücrenin 6 zaman adımında tekrar çağrılması")

print("""
RNN.inner_loop() içinde step() adında küçük bir fonksiyon tanımlanıyor.
Bu fonksiyon HER ZAMAN ADIMINDA aynı self.cell'i (bizim tek LSTMCell'imizi) çağırıyor:""")
keras_kodu_goster(RNN.inner_loop, baslangic="def step(inputs, states):",
                  bitis="return output, new_states")
print("""
Sonra backend.rnn(step, sequences, initial_state, ...) çağrılıyor. Bu fonksiyon
TensorFlow'un hızlı döngüsünü kullanır ama yaptığı iş şu Python döngüsüyle aynıdır:

    states = initial_state                 # [h_0, c_0] = sıfırlar
    for t in range(6):
        output, states = step(sequences[:, t, :], states)

Şimdi bu döngüyü numpy ile kendimiz yapıp Keras'ın sonucuyla karşılaştıralım:""")

h = np.zeros(64, dtype=np.float32)
c = np.zeros(64, dtype=np.float32)
tum_h_elle = []
for t in range(SEQUENCE_LEN):                       # t = 0..5 → saat 1..6
    h, c, _ = hucre_bir_adim(pencere[t], h, c, W, U, b, 64)   # AYNI W, U, b
    tum_h_elle.append(h)
    print(f"  t={t+1}: girdi x_{t+1} (23) + h_{t} (64) → h_{t+1}[:3] = {h[:3]}")
tum_h_elle = np.stack(tum_h_elle)                   # (6, 64)

# Keras'ın kendi hesabı: lstm1'e pencereyi veriyoruz, initial_state VERMİYORUZ.
# pencere[None, ...] → başa batch boyutu ekler: (6, 23) → (1, 6, 23)
tum_h_keras = lstm1(pencere[None, ...]).numpy()[0]  # (6, 64)

fark = np.abs(tum_h_elle - tum_h_keras).max()
print(f"\nNumpy (h_0=0 ile) vs Keras, 6×64 değerin en büyük farkı: {fark:.2e}")
print("→ Aynı. Keras da initial_state verilmeyince sıfırla başlıyor ve aynı hücreyi 6 kez çalıştırıyor."
      if fark < 1e-4 else "→ FARKLI! (beklenmeyen durum)")

# Karşı deney: sıfır yerine 1'lerle başlatsaydık sonuç değişir miydi?
birler = [tf.ones((1, 64)), tf.ones((1, 64))]
tum_h_birler = lstm1(pencere[None, ...], initial_state=birler).numpy()[0]
print(f"Karşı deney: initial_state = 1'ler verilseydi fark: "
      f"{np.abs(tum_h_birler - tum_h_keras).max():.4f}  → sonuç değişirdi.")


# =============================================================================
# BÖLÜM 7 — return_sequences: 6 çıktı mı, 1 çıktı mı?
# =============================================================================
baslik("BÖLÜM 7 — return_sequences=True / False")

print("""
Döngü bittikten sonra RNN.call() hangisini döndüreceğine şu satırlarla karar veriyor:
  outputs     → 6 adımın h'lerinin hepsi
  last_output → sadece son adımın h'si""")
keras_kodu_goster(RNN.call, baslangic="if self.return_sequences:", satir_sayisi=4)

cikis1 = lstm1(X[:32])            # ilk 32 pencere
cikis2 = lstm2(cikis1)
print(f"""
Bizim modelde (ilk 32 pencere ile):
  Girdi                          : {X[:32].shape}  (32 pencere, 6 saat, 23 özellik)
  LSTM-1 (return_sequences=True) : {tuple(cikis1.shape)}  ← h_1..h_6 hepsi, 64'er
  LSTM-2 (return_sequences=False): {tuple(cikis2.shape)}     ← sadece h_6, 32'lik

LSTM-1'in çıktısı LSTM-2'nin girdisi oluyor. Bu yüzden LSTM-2'nin kernel'i
(64, 128): girdi_boyutu=64 (LSTM-1'in units'i), units*4 = 32*4 = 128.
LSTM-2 de aynı şekilde kendi h_0=c_0=0 (32'lik) ile başlayıp 6 adım döner.""")


# =============================================================================
# BÖLÜM 8 — Dropout, BatchNormalization, Dense
# =============================================================================
baslik("BÖLÜM 8 — LSTM'den sonraki katmanlar")

print("""
Dropout(0.2): Eğitim sırasında gelen değerlerin rastgele %20'sini 0 yapar, model
tek bir değere aşırı güvenmesin diye. Tahmin sırasında HİÇBİR ŞEY yapmaz:""")
keras_kodu_goster(Dropout.call, satir_sayisi=4)

ornek = tf.ones((1, 10))
print(f"  training=False: {dropout1(ornek, training=False).numpy()[0]}")
print(f"  training=True : {dropout1(ornek, training=True).numpy()[0]}")
print("  (training=True'da kalan değerler 1/(1-0.2)=1.25 ile büyütülür ki ortalama değişmesin)")

print("""
Dense(16, activation="relu") ve Dense(1): Klasik tam bağlantılı katman.
Hesap: çıktı = aktivasyon(girdi @ kernel + bias)""")
keras_kodu_goster(Dense.call, baslangic="x = ops.matmul(inputs, self.kernel)",
                  satir_sayisi=5)
dense16, dense1 = model.layers[5], model.layers[6]
print(f"""  Bizim modelde:
    Dense(16).kernel.shape = {tuple(dense16.kernel.shape)}  ← LSTM-2'nin 32'lik h_6'sı → 16 sayı
    Dense(1).kernel.shape  = {tuple(dense1.kernel.shape)}   ← 16 sayı → 1 tahmin (congestion_ratio)

BatchNormalization: Gelen 32 sayıyı, eğitim boyunca öğrendiği ortalama ve
standart sapmayla normalize eder; eğitimi daha kararlı hale getirir.""")


# =============================================================================
# BÖLÜM 9 — batch_size=32 eğitimde nasıl uygulanıyor?
# =============================================================================
baslik("BÖLÜM 9 — batch_size=32: model.fit pencereleri nasıl gruplandırıyor?")

n_train = int(len(X) * TRAIN_RATIO)
X_train, y_train = X[:n_train], y[:n_train]
print(f"Bizim split_data() mantığıyla train örnek sayısı: int({len(X)} × {TRAIN_RATIO}) = {n_train}")
print(f"Beklenen batch sayısı: {n_train} / 32 → yukarı yuvarla = {int(np.ceil(n_train / 32))}")
print(f"Batch boyutları: {[min(32, n_train - s) for s in range(0, n_train, 32)]}")


class BatchIzleyici(tf.keras.callbacks.Callback):
    """
    Keras callback'i: model.fit'in her batch'in başında ve sonunda çağırdığı
    fonksiyonları yazmamızı sağlayan sınıf. Burada her batch'ten önce ve sonra
    LSTM-1'in W matrisini kopyalayıp ağırlığın değişip değişmediğine bakıyoruz.
    """
    def on_train_batch_begin(self, batch, logs=None):
        self.once = lstm1.cell.kernel.numpy().copy()

    def on_train_batch_end(self, batch, logs=None):
        sonra = lstm1.cell.kernel.numpy()
        degisim = np.abs(sonra - self.once).max()
        print(f"  Batch {batch + 1}: bitti → LSTM-1 kernel'inde en büyük değişim = {degisim:.6f}")


print("\nGerçek model.fit ile 1 epoch (bizim build_model'in kurduğu modelle):")
model.fit(X_train, y_train, epochs=1, batch_size=32, shuffle=True,
          verbose=0, callbacks=[BatchIzleyici()])
print("""
→ Her batch sonunda ağırlıklar 1 kez değişiyor (32 pencerenin hata ortalamasına göre).
→ Son batch daha küçük (kalan örnekler).
→ Her batch'te RNN.call() yeniden çalıştığı için (Bölüm 4) her pencere h_0=0 ile başlıyor.
→ shuffle=True: her epoch başında pencerelerin SIRASI karışır, pencerelerin
  İÇİNDEKİ 6 saatin sırası değişmez.""")

print("\n\n[TAMAMLANDI] Not: Model eğitilmediği için ağırlıklar rastgele; sayısal değerler")
print("anlamsız, ama boyutlar ve hesap akışı eğitilmiş modelde de birebir aynı.")