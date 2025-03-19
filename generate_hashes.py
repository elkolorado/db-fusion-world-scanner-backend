import cv2
import os
import json
import numpy as np

# Path to the folder containing reference images
REFERENCE_IMAGES_FOLDER = "cards"  # Replace with your folder path
OUTPUT_JSON = "features_db.json"

def extract_features(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    # Initialize ORB detector (or use SIFT if preferred)
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None:
        print(f"No descriptors found for image: {image_path}")
        return None

    # Convert descriptors to a list for JSON serialization
    return descriptors.tolist()

def process_reference_images(folder_path):
    features_db = []

    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
            image_path = os.path.join(folder_path, filename)
            descriptors = extract_features(image_path)

            if descriptors:
                features_db.append({
                    "name": os.path.splitext(filename)[0],  # Use the filename (without extension) as the card name
                    "descriptors": descriptors
                })

    return features_db

if __name__ == "__main__":
    # Process all reference images
    features_db = process_reference_images(REFERENCE_IMAGES_FOLDER)

    # Save the features to a JSON file
    with open(OUTPUT_JSON, "w") as f:
        json.dump(features_db, f, indent=4)

    print(f"Features saved to {OUTPUT_JSON}")