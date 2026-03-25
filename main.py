import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs
import time

# 1. Tarayıcı Ayarları
chrome_options = Options()
# chrome_options.add_argument("--headless") 

# 2. Tarayıcıyı Başlat
driver = webdriver.Chrome(options=chrome_options)

# Verileri biriktireceğimiz ana liste
tum_etkinlikler = []

try:
    url = 'https://www.biletix.com/performance/5SK01/001/TURKIYE/tr'
    print("Sayfa yükleniyor, lütfen bekleyin...")
    
    driver.get(url)
    time.sleep(7) 

    # 4. Sayfa Kaynağını Al ve BeautifulSoup ile İşle
    html_icerigi = driver.page_source
    soup = bs(html_icerigi, 'html.parser')

    # --- VERİ ÇEKME VE DICTIONARY OLUŞTURMA ---
    try:
        # Biletix'in yapısına göre seçicileri (class isimlerini) güncelledim
        etkinlik_adi = soup.find("h1").text.strip() if soup.find("h1") else "Bulunamadı"
        
        # Mekan ve Tarih bilgilerini çekelim (Biletix'te genelde belirli class'lar altındadır)
        # Not: Sitedeki class isimleri değişirse buraları güncellememiz gerekebilir.
        tarih = soup.select_one(".event-date").text.strip() if soup.select_one(".event-date") else "Tarih Yok"
        mekan = soup.select_one(".event-venue").text.strip() if soup.select_one(".event-venue") else "Mekan Yok"
        
        # SÖZLÜK (DICTIONARY) BURADA OLUŞUYOR
        veri_sozlugu = {
            "Etkinlik_Adi": etkinlik_adi,
            "Tarih": tarih,
            "Mekan": mekan,
            "URL": url,
            "Kayit_Zamani": time.ctime() # Verinin ne zaman çekildiğini not etmek iyi bir pratiktir
        }

        # Sözlüğü ana listeye ekliyoruz
        tum_etkinlikler.append(veri_sozlugu)
        print(f"Veri başarıyla çekildi: {etkinlik_adi}")

    except Exception as e:
        print(f"Veri ayrıştırılırken hata oluştu: {e}")

    # --- PANDAS VE PARQUET KAYDETME ---
    if tum_etkinlikler:
        df = pd.DataFrame(tum_etkinlikler)
        
        # Dosyayı kaydet
        df.to_parquet('tübitak_etkinlik_verisi.parquet', engine='pyarrow')
        print("\n--- İŞLEM TAMAM ---")
        print("Veri 'tübitak_etkinlik_verisi.parquet' dosyasına kaydedildi.")
        print(df) # Tabloyu terminalde göster

finally:
    driver.quit()