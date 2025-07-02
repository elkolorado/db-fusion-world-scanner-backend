import sqlite3
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

sift = cv2.SIFT_create()

def extract_keypoints(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def save_keypoints_to_db(image_path, db_path="cards.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    filename = os.path.basename(image_path)
    cursor.execute("SELECT id FROM cards WHERE filename = ?", (filename,))
    result = cursor.fetchone()
    if result:
        card_id = result[0]
    else:
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

def process_image(filename):
    if filename.endswith(".webp"):
        image_path = os.path.join(cards_directory, filename)
        save_keypoints_to_db(image_path)

cards_directory = "cards"
filenames = [f for f in os.listdir(cards_directory) if f.endswith(".webp")]

with ThreadPoolExecutor() as executor:
    executor.map(process_image, filenames)
