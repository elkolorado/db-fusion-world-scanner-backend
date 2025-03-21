import requests

url = "http://127.0.0.1:8000/matchCard"
file_path = r"G:\github\db-fusion-world-img-hashes\WIN_20250315_14_35_06_Pro.jpg"

for _ in range(100):
    with open(file_path, 'rb') as file:
        response = requests.post(url, files={"file": file})
        print(response.status_code, response.text)
