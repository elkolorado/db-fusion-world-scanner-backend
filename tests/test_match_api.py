import unittest
import requests
import os
import argparse
import sys

# Parse -tcg argument (default riftbound)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-tcg', choices=['riftbound', 'fusion'], default='riftbound')
args, remaining_argv = parser.parse_known_args()
tcg = args.tcg

# Define URL and images directory depending on tcg
url = "http://127.0.0.1:8002/matchCard"
if tcg == 'riftbound':
    imgs_dir = os.path.join("tests", "imgs", "riftbound")
else:  # 'fusion'
    imgs_dir = os.path.join("tests", "imgs", "dragon_ball_fusion_world")

if not os.path.isdir(imgs_dir):
    raise RuntimeError(f"Images directory not found: {imgs_dir}")

def generate_test(img_filename):
    def test(self):
        file_path = os.path.join(imgs_dir, img_filename)
        with open(file_path, 'rb') as file:
            response = requests.post(url, files={"file": file})

        print(f"{img_filename} -> {response.json()}")
        self.assertEqual(response.status_code, 200)
        expected_name = os.path.splitext(img_filename)[0]
        self.assertEqual(response.json().get("best_match").split(".")[0], expected_name)
    return test

class TestMatchCardAPI(unittest.TestCase):
    pass

# Dynamically add a test method for each image file
for filename in os.listdir(imgs_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        test_name = f"test_match_card_api_{os.path.splitext(filename)[0]}"
        setattr(TestMatchCardAPI, test_name, generate_test(filename))

if __name__ == '__main__':
    # Remove custom args so unittest doesn't see them
    sys.argv[:] = [sys.argv[0]] + remaining_argv
    unittest.main()