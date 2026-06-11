import requests
import pandas as pd

def open_meteo_cek():
    # Beşiktaş Koordinatları: 41.04, 29.00
    url = "https://api.open-meteo.com/v1/forecast?latitude=41.04&longitude=29.00&hourly=temperature_2m,precipitation,weathercode&windspeed_unit=kmh&timezone=Europe%2FIstanbul"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['hourly']
        df = pd.DataFrame(data)
        
        # Weathercode 51 ve üzeri yağışlı/zorlu koşuldur
        df['Zorlu_Kosul'] = df['weathercode'].apply(lambda x: "Evet" if x >= 51 else "Hayır")
        
        df.to_parquet('besiktas_hava_tahmin_final.parquet', index=False)
        print("Mükemmel! Open-Meteo üzerinden veri çekildi ve dosya oluştu.")
        print(df.head())
    else:
        print("API'ye ulaşılamadı.")

if __name__ == "__main__":
    open_meteo_cek()