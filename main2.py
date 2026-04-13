import time
import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup

def run_scraper():
    print("Tarayıcı başlatılıyor...")
    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options)
    
    try:
        url = "https://www.biletix.com/search/TURKIYE/tr?category_sb=-1&date_sb=-1&city_sb=%C4%B0stanbul"
        driver.get(url)
        time.sleep(12) 
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        events = []
        
        # Biletix'in kart yapısını daha garanti bir yolla bulalım
        items = soup.select('.search-result-item')
        print(f"Sistemde {len(items)} adet etkinlik kartı yakalandı.")

        for item in items:
            try:
                # Kartın içindeki tüm metni alıp parçalara ayırıyoruz
                all_text = item.get_text(separator='|', strip=True)
                parts = all_text.split('|')
                
                # Filtrelemeyi şimdilik kaldırıyoruz, her şeyi çekelim ki mekan isimlerini görelim
                events.append({
                    "Ham_Veri": all_text,
                    "Tahmini_Baslik": parts[0] if len(parts) > 0 else "Bilinmiyor",
                    "Tahmini_Mekan": parts[1] if len(parts) > 1 else "Bilinmiyor"
                })
            except:
                continue

        df = pd.DataFrame(events)
        
        if not df.empty:
            # CSV olarak kaydedelim ki not defteriyle açıp bakabilesin (şimdilik)
            df.to_csv('kontrol_verisi.csv', index=False, encoding='utf-8-sig')
            print(f"Başarılı! {len(df)} etkinlik 'kontrol_verisi.csv' dosyasına yazıldı.")
            print("Lütfen bu dosyayı açıp Beşiktaş mekanlarının orada nasıl yazıldığını kontrol et.")
        else:
            print("Kart bulundu ama içi okunamadı.")

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
    finally:
        # Hata veren quit kısmını deneme yanılma ile geçiyoruz
        try:
            driver.close()
            driver.quit()
        except:
            pass 

if __name__ == "__main__":
    run_scraper()