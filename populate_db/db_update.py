import sqlite3
import cv2
import numpy as np
import os

sift = cv2.SIFT_create()

def extract_keypoints(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def save_keypoints_to_db(image_path, db_path="cards.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    filename = os.path.basename(image_path)

    # Insert image metadata if it doesn't already exist
    cursor.execute("SELECT id FROM cards WHERE filename = ?", (filename,))
    result = cursor.fetchone()
    if result:
        conn.close()
        return  # Already in DB, skip

    cursor.execute("INSERT INTO cards (filename) VALUES (?)", (filename,))
    card_id = cursor.lastrowid

    keypoints, descriptors = extract_keypoints(image_path)
    if descriptors is not None:
        for kp, desc in zip(keypoints, descriptors):
            x, y = kp.pt
            cursor.execute(
                "INSERT INTO keypoints (card_id, x, y, descriptor) VALUES (?, ?, ?, ?)",
                (card_id, x, y, desc.tobytes())
            )

    conn.commit()
    conn.close()

# Get all filenames already in the DB
def get_existing_filenames(db_path="cards.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM cards")
    filenames = set(row[0] for row in cursor.fetchall())
    conn.close()
    return filenames

cards_directory = "cards"
existing_filenames = get_existing_filenames()

for filename in os.listdir(cards_directory):
    if filename.endswith(".webp") and filename not in existing_filenames:
        image_path = os.path.join(cards_directory, filename)
        save_keypoints_to_db(image_path)