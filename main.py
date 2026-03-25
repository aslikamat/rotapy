import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs
import time

# 1. Tarayıcı Ayarları
chrome_options = Options()
# chrome_options.add_argument("--headless") # İstersen görünmez çalıştırabilirsin

driver = webdriver.Chrome(options=chrome_options)
tum_etkinlikler = []

try:
    url = 'https://www.biletix.com/performance/5SK01/001/TURKIYE/tr'
    print("Sayfa yükleniyor...")
    driver.get(url)
    time.sleep(5) # Sayfanın tam yüklenmesi için bekliyoruz

    html_icerigi = driver.page_source
    soup = bs(html_icerigi, 'html.parser')

    # --- HASSAS VERİ ÇEKME ---
    try:
        # Etkinlik Adı (Genelde h1 içindedir)
        etkinlik_adi = soup.find("h1").text.strip() if soup.find("h1") else "Bulunamadı"

        # Yeni Biletix yapısında tarih ve mekan genelde "performance-header" veya benzeri divlerdedir
        # Aşağıdaki seçicileri Biletix'in en güncel (2026) yapısına göre revize ettim:
        tarih = soup.select_one(".event-date, .performance-date, .date").text.strip() if soup.select_one(".event-date, .performance-date, .date") else "Tarih Belirlenmedi"
        
        mekan = soup.select_one(".event-venue, .venue-link, #venue-name").text.strip() if soup.select_one(".event-venue, .venue-link, #venue-name") else "Mekan Belirlenmedi"

        # --- KAPASİTE BİLGİSİ ---
        # Kapasite genelde mekan isminin yanında veya "Mekan Hakkında" kısmında gizlidir.
        # Eğer sayfada "Kapasite" kelimesi geçiyorsa onu bulmaya çalışalım:
        kapasite = "Bilgi Yok"
        sayfa_metni = soup.get_text()
        if "kapasite" in sayfa_metni.lower():
            # Basit bir mantıkla kapasite kelimesinden sonraki sayıları yakalamaya çalışabiliriz
            import re
            match = re.search(r"kapasite[^\d]*(\d+[\d\s\.]*)", sayfa_metni.lower())
            if match:
                kapasite = match.group(1).strip()

        veri_sozlugu = {
            "Etkinlik_Adi": etkinlik_adi,
            "Tarih": tarih,
            "Mekan": mekan,
            "Kapasite": kapasite,
            "URL": url,
            "Kayit_Zamani": time.ctime()
        }

        tum_etkinlikler.append(veri_sozlugu)
        print(f"Başarılı! Çekilen Veri: {etkinlik_adi} | {tarih} | {mekan} | Kapasite: {kapasite}")

    except Exception as e:
        print(f"Veri ayıklanırken bir hata oluştu: {e}")

    # --- KAYDETME ---
    if tum_etkinlikler:
        df = pd.DataFrame(tum_etkinlikler)
        df.to_parquet('tübitak_etkinlik_verisi.parquet', engine='pyarrow')
        print("\nVeri başarıyla kaydedildi.")
        print(df)

finally:
    driver.quit()