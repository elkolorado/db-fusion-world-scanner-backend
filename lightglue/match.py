import sqlite3
import cv2
import numpy as np

# ORB Detector
orb = cv2.ORB_create(nfeatures=500)

def extract_keypoints(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_card(query_image_path, db_path="cards.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query_keypoints, query_descriptors = extract_keypoints(query_image_path)

    # BFMatcher (Brute-Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    best_match = None
    best_score = 0

    cursor.execute("SELECT id, filename FROM cards")
    for card_id, filename in cursor.fetchall():
        cursor.execute("SELECT descriptor FROM keypoints WHERE card_id = ?", (card_id,))
        stored_descriptors = [np.frombuffer(desc, dtype=np.uint8) for (desc,) in cursor.fetchall()]
        
        if stored_descriptors:
            stored_descriptors = np.vstack(stored_descriptors)
            matches = bf.match(query_descriptors, stored_descriptors)
            score = len(matches)

            if score > best_score:
                best_score = score
                best_match = filename

    conn.close()
    return best_match

# Example usage
matched_card = match_card("WIN_20250315_14_35_06_Pro.jpg")
print(f"Best match: {matched_card}")
