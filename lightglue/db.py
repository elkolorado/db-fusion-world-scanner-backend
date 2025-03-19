import sqlite3
import cv2
import numpy as np
import os

sift = cv2.SIFT_create()

def extract_keypoints(image_path):
    """Extract keypoints and descriptors from a full-color image."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Use full-color image
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def save_keypoints_to_db(image_path, db_path="cards.db"):
    """Extract and save keypoints and descriptors to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract filename from the full path
    filename = os.path.basename(image_path)

    # Insert image metadata if it doesn't already exist
    cursor.execute("SELECT id FROM cards WHERE filename = ?", (filename,))
    result = cursor.fetchone()
    if result:
        card_id = result[0]
    else:
        cursor.execute("INSERT INTO cards (filename) VALUES (?)", (filename,))
        card_id = cursor.lastrowid

    # Extract keypoints
    keypoints, descriptors = extract_keypoints(image_path)

    # Insert keypoints
    for kp, desc in zip(keypoints, descriptors):
        x, y = kp.pt
        cursor.execute(
            "INSERT INTO keypoints (card_id, x, y, descriptor) VALUES (?, ?, ?, ?)",
            (card_id, x, y, desc.tobytes())  # Store descriptor as bytes
        )

    conn.commit()
    conn.close()

# Process all files in the "cards" directory
cards_directory = "cards"
for filename in os.listdir(cards_directory):
    if filename.endswith(".webp"):  # Process only .webp files
        image_path = os.path.join(cards_directory, filename)
        save_keypoints_to_db(image_path)