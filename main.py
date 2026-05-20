import requests
import pandas as pd

# TomTom Developer Portal'dan aldığın ücretsiz API anahtarı
API_KEY = "9dqoh1Ho4b8xTvYB3tnZCtGEb5qUS2S1"

# Örnek: Beşiktaş Barbaros Bulvarı üzerindeki bir koordinat
LAT = "41.0422"
LON = "29.0083"

def get_traffic_density(lat, lon, api_key):
    # Traffic Flow Segment Data uç noktası
    # style: 10 (absolute values), zoom: 10 (genel yol ağları)
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={api_key}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        flow_data = data.get('flowSegmentData', {})
        
        # Mevcut hız ve serbest akış (trafiksiz) hızı
        current_speed = flow_data.get('currentSpeed')
        free_flow_speed = flow_data.get('freeFlowSpeed')
        
        # Yoğunluk hesaplama (Hız ne kadar düştüyse trafik o kadar yoğundur)
        # Oran 1'e yakınsa trafik yok, 0'a yakınsa trafik kilitli.
        congestion_ratio = current_speed / free_flow_speed if free_flow_speed else 1
        
        return {
            "current_speed_kmh": current_speed,
            "free_flow_speed_kmh": free_flow_speed,
            "congestion_ratio": round(congestion_ratio, 2)
        }
    else:
        print(f"Hata Kodu: {response.status_code}")
        return None

# Fonksiyonu çalıştır
traffic_status = get_traffic_density(LAT, LON, API_KEY)
print("Anlık Trafik Durumu:", traffic_status)