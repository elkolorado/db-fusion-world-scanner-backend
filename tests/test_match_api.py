import unittest
import requests
import os

# Define constants directly in the test file
url = "http://127.0.0.1:8002/matchCard"
imgs_dir = "tests/imgs"
def generate_test(img_filename):
    def test(self):
        file_path = os.path.join(imgs_dir, img_filename)
        with open(file_path, 'rb') as file:
            response = requests.post(url, files={"file": file})

        print(f"{img_filename} -> {response.json()}")
        self.assertEqual(response.status_code, 200)
        expected_name = os.path.splitext(img_filename)[0] + ".webp"
        self.assertEqual(response.json().get("best_match"), expected_name)
    return test

class TestMatchCardAPI(unittest.TestCase):
    pass

# Dynamically add a test method for each image file
for filename in os.listdir(imgs_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        test_name = f"test_match_card_api_{os.path.splitext(filename)[0]}"
        setattr(TestMatchCardAPI, test_name, generate_test(filename))

if __name__ == '__main__':
    unittest.main()