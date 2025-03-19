import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor

# Load sets from cards_urls/sets.json
SETS_FILE = 'cards_urls/sets.json'
CARDS_FOLDER = 'cards'

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def process_json_file(json_file):
    with open(json_file, 'r') as file:
        urls = json.load(file)
    for url in urls:
        filename = os.path.basename(url)
        save_path = os.path.join(CARDS_FOLDER, filename)
        download_image(url, save_path)

def main():
    # Ensure the cards folder exists
    os.makedirs(CARDS_FOLDER, exist_ok=True)

    # Load the sets.json file
    with open(SETS_FILE, 'r') as file:
        json_files = json.load(file)

    # Prepend the directory path to each JSON file
    json_files = [os.path.join('cards_urls', json_file) for json_file in json_files]

    # Process each JSON file in parallel with a max of 5 threads
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(process_json_file, json_files)

if __name__ == '__main__':
    main()