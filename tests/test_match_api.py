import unittest
import requests

# Define constants directly in the test file
url = "http://127.0.0.1:8000/matchCard"
file_path = "tests/WIN_20250315_14_35_06_Pro.jpg"

class TestMatchCardAPI(unittest.TestCase):
    def test_match_card_api(self):
        # Perform the POST request
        with open(file_path, 'rb') as file:
            response = requests.post(url, files={"file": file})

        print(response.json())
        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"best_match": "FB05-107.webp"})

if __name__ == '__main__':
    unittest.main()