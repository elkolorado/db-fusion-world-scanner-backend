from fastapi import FastAPI, File, UploadFile
import sqlite3
import cv2
import numpy as np
import os
import pickle
import time
import faiss

from concurrent.futures import ThreadPoolExecutor


app = FastAPI()

# ORB Detector
sift = cv2.SIFT_create(nfeatures=500)


# File paths for saved descriptors and FAISS index
DESCRIPTORS_FILE = "descriptors.pkl"
FAISS_FILE = "faiss_index.bin"


def extract_keypoints_from_bytes(image_bytes):
    """Extract keypoints and descriptors from image bytes using full-color processing."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # Use full-color image
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def load_descriptors(db_path="cards.db"):
    """Load all descriptors and filenames into memory."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query all descriptors and filenames
    cursor.execute("""
        SELECT cards.filename, keypoints.descriptor
        FROM cards
        JOIN keypoints ON cards.id = keypoints.card_id
    """)
    rows = cursor.fetchall()

    # Group descriptors by filename
    descriptors_by_filename = {}
    for filename, descriptor in rows:
        if filename not in descriptors_by_filename:
            descriptors_by_filename[filename] = []
        descriptors_by_filename[filename].append(np.frombuffer(
            descriptor, dtype=np.float32))  # Use float32 for SIFT

    conn.close()
    print(
        f"Loaded {len(descriptors_by_filename)} descriptors from the database")
    return descriptors_by_filename


# Precompute FAISS index for all descriptors


def precompute_faiss_index():
    """Precompute FAISS index for all descriptors."""
    all_descriptors = []
    filenames = []

    # Collect all descriptors and filenames
    for filename, descriptors in DESCRIPTORS_CACHE.items():
        all_descriptors.append(np.vstack(descriptors))
        # Map each descriptor to its filename
        filenames.extend([filename] * len(descriptors))

    # Stack all descriptors into a single NumPy array
    all_descriptors = np.vstack(all_descriptors).astype(np.float32)

    # Create a FAISS index
    # Descriptor dimensionality (128 for SIFT)
    dimension = all_descriptors.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
    index.add(all_descriptors)  # Add descriptors to the index

    print(f"FAISS index built with {index.ntotal} descriptors.")
    return index, filenames


FILENAMES_FILE = "filenames.pkl"

print("Loading FAISS index...")
if os.path.exists(FAISS_FILE) and os.path.exists(FILENAMES_FILE):
    # Load FAISS index from file
    FAISS_INDEX = faiss.read_index(FAISS_FILE)
    print(f"FAISS index loaded from {FAISS_FILE}")

    # Load FILENAMES from file
    with open(FILENAMES_FILE, "rb") as f:
        FILENAMES = pickle.load(f)
    print(f"Filenames loaded from {FILENAMES_FILE}")
else:
    # Compute descriptors and FAISS index from scratch
    DESCRIPTORS_CACHE = load_descriptors()  # Load descriptors from the database
    FAISS_INDEX, FILENAMES = precompute_faiss_index()

    # Save FAISS index to file
    faiss.write_index(FAISS_INDEX, FAISS_FILE)
    print(f"FAISS index saved to {FAISS_FILE}")

    # Save FILENAMES to file
    with open(FILENAMES_FILE, "wb") as f:
        pickle.dump(FILENAMES, f)
    print(f"Filenames saved to {FILENAMES_FILE}")


def match_card(image_bytes):
    """Match the query image using the FAISS index with GPU acceleration and parallel processing."""
    _, query_descriptors = extract_keypoints_from_bytes(image_bytes)

    # Convert query descriptors to NumPy array
    query_descriptors = np.array(query_descriptors, dtype=np.float32)

    # Check if GPU FAISS is available and use it
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()  # Initialize GPU resources
        dimension = query_descriptors.shape[1]
        gpu_index = faiss.index_cpu_to_gpu(
            res, 0, FAISS_INDEX)  # Transfer index to GPU
    else:
        gpu_index = FAISS_INDEX

    # Perform nearest neighbor search
    k = 2  # Number of nearest neighbors
    distances, indices = gpu_index.search(query_descriptors, k)

    # Apply Lowe's ratio test in parallel
    def process_match(i):
        if distances[i][0] < 0.7 * distances[i][1]:  # Lowe's ratio test
            return indices[i][0]
        return None

    with ThreadPoolExecutor() as executor:
        good_matches = list(filter(None, executor.map(
            process_match, range(len(distances)))))

    # Count matches for each filename
    match_counts = {}
    for match_idx in good_matches:
        filename = FILENAMES[match_idx]
        match_counts[filename] = match_counts.get(filename, 0) + 1

    # Find the best match
    best_match = max(match_counts, key=match_counts.get, default=None)
    return best_match

@app.post("/matchCard")
async def match_card_api(file: UploadFile = File(...)):
    """API endpoint to match a card."""
    image_bytes = await file.read()

    # Save the uploaded image to a file (optional, for debugging)
    # with open("uploaded_image.jpg", "wb") as f:
    #     f.write(image_bytes)

    # Match the card
    best_match = match_card(image_bytes)

    # Return the best match
    return {"best_match": best_match}

#file response
from fastapi.responses import FileResponse

@app.get("/displayImage")
async def display_image(filename: str):
    """Serve an image from the cards folder."""
    file_path = f"cards/{filename}"
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(file_path)



@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"Hello": "World"}

import asyncio
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

@app.get("/cardInfo")
async def get_card_info(cardname: str):
    """Get card info from CardMarket using Selenium."""
    def fetch_card_info():
        # Set up Selenium WebDriver
        chrome_options = Options()
        # chrome_options.add_argument("--headless")  # Run in headless mode
        # chrome_options.add_argument("--disable-gpu")
        # chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=chrome_options)

        try:
            # Navigate to the CardMarket search page
            url = f"https://www.cardmarket.com/en/DragonBallSuper/Products/Search?searchString={cardname.split('_')[0]}"
            driver.get(url)
            card_prefix = cardname.split("-")[0].split("_")[0].upper()

            # Wait for the table to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "table-body"))
            )

            # Find all rows in the table
            table_body = driver.find_element(By.CLASS_NAME, "table-body")
            rows = [row for row in table_body.find_elements(By.CLASS_NAME, "row.g-0") if "productRow" in row.get_attribute("id")]

            # Parse the rows to find the matching card
            for row in rows:
                try:
                    # Extract the expansion (e.g., FB03, FWUP)
                    expansion_element = row.find_element(By.CLASS_NAME, "col-icon.small")
                    expansion = expansion_element.find_element(By.TAG_NAME, "span").text.strip()

                    # Check if the expansion matches the card prefix
                    if expansion == card_prefix:
                        # Extract card details
                        card_name = row.find_element(By.TAG_NAME, "a").text.strip()
                        card_number = row.find_element(By.CLASS_NAME, "col-number").text.strip()
                        availability = row.find_element(By.CLASS_NAME, "col-availability").text.strip()
                        price = row.find_element(By.CLASS_NAME, "col-price").text.strip()
                        link = row.find_element(By.CLASS_NAME, "col-12.col-md-8.px-2.flex-column").find_element(By.TAG_NAME, "a").get_attribute("href")
                        # Return the matching card details
                        return {
                            "name": cardname,
                            "number": card_number,
                            "availability": availability,
                            "price": price,
                            "expansion": expansion,
                            "link": link
                        }
                except Exception as e:
                    # Skip rows with missing data
                    print(f"Error processing row: {e}")

            # If no matching card is found
            return {"error": "No matching card found"}

        finally:
            driver.quit()

    # Run the blocking Selenium operation in a separate thread
    return await asyncio.to_thread(fetch_card_info)

