import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs
import time
from datetime import datetime, timedelta

def gecmis_hava_durumu_cek():
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    driver = webdriver.Chrome(options=chrome_options)
    
    # İstanbul için detaylı geçmiş veri sunan güvenilir bir kaynak
    url = "https://www.timeanddate.com/weather/turkey/istanbul/historic"
    
    try:
        print("Son 2 saatin verileri analiz ediliyor...")
        driver.get(url)
        time.sleep(5)
        
        soup = bs(driver.page_source, 'html.parser')
        
        # Tablodaki geçmiş saat verilerini bulalım
        # Bu sitede 'wt-his' ID'li tabloda saatlik geçmiş veriler yer alır
        tablo = soup.find("table", {"id": "wt-his"})
        satirlar = tablo.find("tbody").find_all("tr")
        
        gecmis_veriler = []
        simdi = datetime.now()
        iki_saat_once = simdi - timedelta(hours=2)

        for satir in satirlar:
            hucreler = satir.find_all(["td", "th"])
            if len(hucreler) > 5:
                # Sitedeki saat formatı genelde "21:20" veya "09:50 PM" gibidir
                saat_metni = hucreler[0].get_text(strip=True)
                
                # Saati bugünün tarihiyle birleştirip kontrol edelim
                # Not: Bu kısım sitenin o anki saat formatına göre ufak düzenleme isteyebilir
                try:
                    veri_saati = datetime.strptime(saat_metni, "%H:%M").replace(
                        year=simdi.year, month=simdi.month, day=simdi.day
                    )
                except:
                    continue

                # Sadece son 2 saati filtrele
                if veri_saati >= iki_saat_once:
                    derece = hucreler[1].get_text(strip=True).replace("°C", "")
                    durum = hucreler[2].get_text(strip=True).lower()
                    ruzgar = hucreler[3].get_text(strip=True)
                    
                    # --- ZORLU KOŞUL TESPİTİ ---
                    zorlu_kelimeler = ["rain", "storm", "wind", "yağmur", "fırtına", "sağanak"]
                    is_zorlu = any(k in durum for k in zorlu_kelimeler)
                    
                    gecmis_veriler.append({
                        "Veri_Saati": veri_saati.strftime("%Y-%m-%d %H:%M"),
                        "Derece": derece,
                        "Hava_Durumu": durum,
                        "Ruzgar_Hizi": ruzgar,
                        "Zorlu_Kosul": "Evet" if is_zorlu else "Hayır",
                        "Konum": "Beşiktaş/İstanbul"
                    })

        if gecmis_veriler:
            df = pd.DataFrame(gecmis_veriler)
            # Parquet olarak kaydet
            df.to_parquet('besiktas_gecmis_2saat.parquet', engine='pyarrow')
            print("\n--- GEÇMİŞ VERİ KAYDEDİLDİ ---")
            print(df)
        else:
            print("Son 2 saate ait veri bulunamadı. Tabloyu kontrol edin.")

    finally:
        driver.quit()

if __name__ == "__main__":
    gecmis_hava_durumu_cek()
    