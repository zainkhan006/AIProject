import os
import time
import requests
import pandas as pd

csv_path = r'C:\Users\Zain Ul Ibad\Desktop\aistuff\MovieGenre.csv'

df = pd.read_csv(csv_path, encoding='ISO-8859-1')

download_dir = r'C:\Users\Zain Ul Ibad\Desktop\aistuff\images'
os.makedirs(download_dir, exist_ok=True)

for idx, row in df.iterrows():
    url = row['Poster']
    movie_id = row['imdbId']

    if isinstance(url, str) and url.startswith('http'):
        save_path = os.path.join(download_dir, f"{movie_id}.jpg")
        try:
            head_resp = requests.head(url, allow_redirects=True, timeout=5)
            if head_resp.status_code != 200:
                print(f"Skipping {movie_id}: HEAD request returned status {head_resp.status_code}")
                continue
        except requests.RequestException as e:
            print(f"Skipping {movie_id}: HEAD request failed with error {e}")
            continue
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(resp.content)
                    print(f"Downloaded {movie_id}")
                    break  
                elif resp.status_code == 404:
                    print(f"Failed {movie_id}: 404 Not Found")
                    break  
                else:
                    print(f"Attempt {attempt + 1} for {movie_id} returned status {resp.status_code}")
            except requests.RequestException as e:
                print(f"Attempt {attempt + 1} for {movie_id} failed with error: {e}")
            time.sleep(2 ** attempt) 
