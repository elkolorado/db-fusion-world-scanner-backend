from fastapi import FastAPI, File, UploadFile
import sqlite3
import cv2
import numpy as np
import os
import pickle
import time
import faiss
import jwt
from fastapi import HTTPException, Depends
from datetime import datetime, timedelta


from pydantic import BaseModel
import httpx

from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "http://127.0.0.1"],  # Allow localhost
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ORB Detector
sift = cv2.SIFT_create(nfeatures=1000)


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
    """Precompute FAISS index for all descriptors using IndexIVFPQ for compression."""
    all_descriptors = []
    filenames = []

    # Collect all descriptors and filenames
    for filename, descriptors in DESCRIPTORS_CACHE.items():
        all_descriptors.append(np.vstack(descriptors))
        filenames.extend([filename] * len(descriptors))

    all_descriptors = np.vstack(all_descriptors).astype(np.float32)
    dimension = all_descriptors.shape[1]

    # Use IVFPQ index for compression, tuned for better accuracy
    nlist = 1024  # number of clusters (higher for better recall)
    m = 16        # number of subquantizers (higher for better accuracy)
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)  # 8 bits per code
    print(f"Training IVFPQ index with nlist={nlist}, m={m}...")
    index.train(all_descriptors)
    print("Adding descriptors to IVFPQ index...")
    index.add(all_descriptors)
    print(f"FAISS IVFPQ index built with {index.ntotal} descriptors.")
    return index, filenames


FILENAMES_FILE = "filenames.pkl"

def build_faiss_from_images(cards_dir="cards"):
    """Build FAISS index and filenames directly from images in the cards directory."""
    all_descriptors = []
    filenames = []
    for filename in os.listdir(cards_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            file_path = os.path.join(cards_dir, filename)
            image = cv2.imread(file_path)
            if image is None:
                print(f"Warning: Could not read {file_path}")
                continue
            keypoints, descriptors = sift.detectAndCompute(image, None)
            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.append(descriptors)
                filenames.extend([filename] * len(descriptors))
    if not all_descriptors:
        raise RuntimeError("No descriptors found in any image.")
    all_descriptors = np.vstack(all_descriptors).astype(np.float32)
    dimension = all_descriptors.shape[1]
    nlist = 4096  # Increase clusters for better recall
    m = 32        # Increase subquantizers for better accuracy
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
    print(f"Training IVFPQ index with nlist={nlist}, m={m}...")
    index.train(all_descriptors)
    print("Adding descriptors to IVFPQ index...")
    index.add(all_descriptors)
    print(f"FAISS IVFPQ index built with {index.ntotal} descriptors.")
    return index, filenames

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
    # Build descriptors and FAISS index directly from images
    FAISS_INDEX, FILENAMES = build_faiss_from_images("cards")
    faiss.write_index(FAISS_INDEX, FAISS_FILE)
    print(f"FAISS index saved to {FAISS_FILE}")
    with open(FILENAMES_FILE, "wb") as f:
        pickle.dump(FILENAMES, f)
    print(f"Filenames saved to {FILENAMES_FILE}")

# Step 2: Create GPU index ONCE
if faiss.get_num_gpus() > 0:
    FAISS_RES = faiss.StandardGpuResources()
    GPU_INDEX = faiss.index_cpu_to_gpu(FAISS_RES, 0, FAISS_INDEX)
else:
    GPU_INDEX = FAISS_INDEX

print(f"FAISS GPU available: {faiss.get_num_gpus() > 0}")
print(f"FAISS index type: {type(FAISS_INDEX)}")
print(f"FAISS index size: {FAISS_INDEX.ntotal}")


def match_card(image_bytes):
    """Match the query image using the FAISS index with GPU acceleration and parallel processing."""
    start_time = time.time()
    _, query_descriptors = extract_keypoints_from_bytes(image_bytes)
    extract_time = time.time()
    print(f"Time to extract keypoints: {extract_time - start_time:.4f} seconds")

    # Convert query descriptors to NumPy array
    query_descriptors = np.array(query_descriptors, dtype=np.float32)

    # Perform nearest neighbor search
    search_start = time.time()
    k = 2
    gpu_start = time.time()
    if faiss.get_num_gpus() > 0:
        print("Using FAISS GPU for search")
        res = faiss.StandardGpuResources()  # Initialize GPU resources
        dimension = query_descriptors.shape[1]
        gpu_index = faiss.index_cpu_to_gpu(
            res, 0, FAISS_INDEX)  # Transfer index to GPU
    else:
        print("Using FAISS CPU for search")
        gpu_index = FAISS_INDEX
    gpu_end = time.time()
    print(f"Time to prepare FAISS index: {gpu_end - gpu_start:.4f} seconds")
    distances, indices = gpu_index.search(query_descriptors, k)
    search_end = time.time()
    print(f"Query descriptors shape: {query_descriptors.shape}")
    print(f"Time for FAISS search: {search_end - search_start:.4f} seconds")

    # Apply Lowe's ratio test in parallel
    ratio_start = time.time()
    def process_match(i):
        if distances[i][0] < 0.7 * distances[i][1]:  # Lowe's ratio test
            return indices[i][0]
        return None

    with ThreadPoolExecutor() as executor:
        good_matches = list(filter(None, executor.map(
            process_match, range(len(distances)))))
    ratio_end = time.time()
    print(f"Time for Lowe's ratio test: {ratio_end - ratio_start:.4f} seconds")

    # Count matches for each filename
    count_start = time.time()
    match_counts = {}
    for match_idx in good_matches:
        filename = FILENAMES[match_idx]
        match_counts[filename] = match_counts.get(filename, 0) + 1
    count_end = time.time()
    print(f"Time to count matches: {count_end - count_start:.4f} seconds")

    # Find the best match
    best_match = max(match_counts, key=match_counts.get, default=None)
    total_time = time.time() - start_time
    print(f"Total time in match_card: {total_time:.4f} seconds")
    return best_match

@app.post("/matchCard")
async def match_card_api(file: UploadFile = File(...)):
    """API endpoint to match a card."""
    image_bytes = await file.read()

    # Save the uploaded image to a file (optional, for debugging)
    with open("uploaded_image.jpg", "wb") as f:
        f.write(image_bytes)

    # Match the card
    best_match = match_card(image_bytes)

    if best_match:
        card_market_id = os.path.splitext(best_match)[0]
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:8000/card/{card_market_id}")
            if response.status_code == 200:
                card_details = response.json()
                return {"best_match": best_match, "card_details": card_details}
            else:
                return {"best_match": best_match, "error": "Failed to fetch card details"}
    else:
        return {"best_match": None, "error": "No match found"}

    # Return the best match
    return {"best_match": best_match}

#file response

from fastapi.responses import FileResponse


# New endpoint: get card image by cardMarketId and extension
@app.get("/card-image/{cardMarketId}.{extension}")
async def get_card_image(cardMarketId: str, extension: str):
    """Serve a card image from the cards folder by cardMarketId and extension."""
    filename = f"{cardMarketId}.{extension}"
    file_path = os.path.join("cards", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Card image not found")
    return FileResponse(file_path)



@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"Hello": "World"}




# Secret key for signing JWTs
SECRET_KEY = "your_secret_key2"
ALGORITHM = "HS256"

import bcrypt

def hash_password(password: str) -> str:
    """Hash a plaintext password."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate the user by checking the username and hashed password in the database."""
    conn = sqlite3.connect("cards.db")
    cursor = conn.cursor()

    # Use parameterized query to prevent SQL injection
    query = """
        SELECT password
        FROM users
        WHERE username = ?
    """
    cursor.execute(query, (username,))
    row = cursor.fetchone()
    conn.close()

    # Check if the user exists and the password matches
    if row and bcrypt.checkpw(password.encode('utf-8'), row[0].encode('utf-8')):
        return True
    return False

from pydantic import BaseModel
# Define a Pydantic model for the login request
class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
async def login(request: LoginRequest):
    """Authenticate user and return a JWT token."""
    username = request.username
    password = request.password
    if not authenticate_user(username, password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Generate JWT token
    expiration = datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour
    payload = {
        "sub": username,
        "exp": expiration
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return {"token": token, "token_type": "bearer"}


from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from typing import List

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Define the Card interface
class Card(BaseModel):
    id: str
    name: str
    image: str
    quantity: int
    set: str

def decode_jwt(token: str):
    """Decode and verify the JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Extract the current user from the JWT token."""
    payload = decode_jwt(token)
    return payload.get("sub")  # Return the username (subject)

@app.get("/collection", response_model=List[Card])
async def get_collection(current_user: str = Depends(get_current_user)):
    """Fetch the user's card collection."""
    # Connect to the database
    conn = sqlite3.connect("cards.db")
    cursor = conn.cursor()

    # Query the user's collection securely using parameterized query
    cursor.execute("""
        SELECT cards.id, cards.filename, cards.name, user_cards.quantity, cards."set"
        FROM user_cards
        JOIN cards ON user_cards.card_id = cards.id
        WHERE user_cards.username = ?
    """, (current_user,))
    rows = cursor.fetchall()
    conn.close()

    # Format the result as a list of Card objects
    collection = [
        Card(
            id=str(row[0]) if row[0] is not None else "",
            name=row[1] if row[1] is not None else "",
            image=f"/displayImage?filename={row[1]}" if row[1] is not None else "",
            quantity=row[3] if row[3] is not None else 0,
            set=row[4] if row[4] is not None else ""
        )
        for row in rows
    ]

    return collection


# Define a Pydantic model for adding a card

@app.post("/collection/add")
async def add_card_to_collection(request: Card, current_user: str = Depends(get_current_user)):
    print(request)
    """Add a card to the user's collection."""
    card_id = request.name + '.webp'

    print(card_id)

    # Connect to the database
    conn = sqlite3.connect("cards.db")
    cursor = conn.cursor()

    # Check if the card exists in the cards table
    cursor.execute("SELECT id FROM cards WHERE filename = ?", (card_id,))
    card = cursor.fetchone()
    currentCardId = card[0] if card else None
    print(card)
    if not card:
        conn.close()
        raise HTTPException(status_code=404, detail="Card not found")

    # Check if the user already has the card in their collection
    cursor.execute("""
        SELECT quantity
        FROM user_cards
        WHERE username = ? AND card_id = ?
    """, (current_user, currentCardId))
    user_card = cursor.fetchone()

    if user_card is not None:
        # Update the quantity if the card already exists in the user's collection
        new_quantity = user_card[0] + 1
        cursor.execute("""
            UPDATE user_cards
            SET quantity = ?
            WHERE username = ? AND card_id = ?
        """, (new_quantity, current_user, currentCardId))
    else:
        # Insert the card into the user's collection
        cursor.execute("""
            INSERT INTO user_cards (username, card_id, quantity)
            VALUES (?, ?, ?)
        """, (current_user, currentCardId, 1))

    conn.commit()
    conn.close()

    return request


# Define a Pydantic model for removing a card
class RemoveCardRequest(BaseModel):
    id: str

@app.post("/collection/removeCard")
async def remove_card_from_collection(request: RemoveCardRequest, current_user: str = Depends(get_current_user)):
    """Remove a single quantity of a card from the user's collection."""
    card_id = request.id + '.webp'

    # Connect to the database
    conn = sqlite3.connect("cards.db")
    cursor = conn.cursor()

    # Check if the card exists in the cards table
    cursor.execute("""
        SELECT user_cards.card_id, user_cards.quantity 
        FROM user_cards 
        JOIN cards ON user_cards.card_id = cards.id 
        WHERE user_cards.username = ? AND cards.filename = ?
    """, (current_user, card_id))
    user_card = cursor.fetchone()

    if not user_card:
        conn.close()
        raise HTTPException(status_code=404, detail="Card not found in user's collection")

    current_card_id = user_card[0]
    current_quantity = user_card[1]

    if current_quantity > 1:
        # Decrease the quantity by 1
        new_quantity = current_quantity - 1
        cursor.execute("""
            UPDATE user_cards SET quantity = ? WHERE username = ? AND card_id = ?
        """, (new_quantity, current_user, current_card_id))
    else:
        # Remove the card from the user's collection if the quantity is 1
        cursor.execute("""
            DELETE FROM user_cards WHERE username = ? AND card_id = ?
        """, (current_user, current_card_id))

    conn.commit()
    conn.close()

    return {"message": "Card quantity updated successfully"}



@app.post("/matchCards")
async def match_cards_api(file: UploadFile = File(...)):
    t0 = time.time()
    image_bytes = await file.read()
    t1 = time.time()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    t2 = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t3 = time.time()
    rects = []
    card_imgs = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 10000:
            x, y, w, h = cv2.boundingRect(approx)
            card_img = image[y:y+h, x:x+w]
            card_imgs.append(card_img)
            _, card_bytes = cv2.imencode('.jpg', card_img)
            card_bytes = card_bytes.tobytes()
            rects.append((approx, x, y, w, h, card_bytes))
    t4 = time.time()
    # --- Batch SIFT extraction ---
    descriptors_list = []
    descriptor_ranges = []
    for card_img in card_imgs:
        keypoints, descriptors = sift.detectAndCompute(card_img, None)
        if descriptors is not None and len(descriptors) > 0:
            descriptor_ranges.append((len(descriptors_list), len(descriptors_list) + len(descriptors)))
            descriptors_list.extend(descriptors)
        else:
            descriptor_ranges.append((len(descriptors_list), len(descriptors_list)))
    t5 = time.time()
    if len(descriptors_list) == 0:
        card_matches = [None] * len(card_imgs)
    else:
        all_descriptors = np.array(descriptors_list, dtype=np.float32)
        k = 2
        gpu_index = GPU_INDEX if faiss.get_num_gpus() > 0 else FAISS_INDEX
        print(f"[DEBUG] faiss.get_num_gpus() = {faiss.get_num_gpus()}")
        print(f"[DEBUG] gpu_index type: {type(gpu_index)}")
        print(f"[DEBUG] all_descriptors shape: {all_descriptors.shape}")
        print(f"[DEBUG] FAISS_INDEX.ntotal: {FAISS_INDEX.ntotal}")
        t6 = time.time()
        distances, indices = gpu_index.search(all_descriptors, k)
        t7 = time.time()
        # --- Lowe's ratio test for all descriptors ---
        good_matches = [None] * len(all_descriptors)
        for i in range(len(all_descriptors)):
            if distances[i][0] < 0.7 * distances[i][1]:
                good_matches[i] = indices[i][0]
        # --- Assign matches to each card by range ---
        card_matches = []
        for start, end in descriptor_ranges:
            match_counts = {}
            for idx in range(start, end):
                match_idx = good_matches[idx]
                if match_idx is not None:
                    filename = FILENAMES[match_idx]
                    match_counts[filename] = match_counts.get(filename, 0) + 1
            best_match = max(match_counts, key=match_counts.get, default=None) if match_counts else None
            card_matches.append(best_match)
    t8 = time.time()
    vis_image = image.copy()
    for (approx, x, y, w, h, _), match in zip(rects, card_matches):
        cv2.drawContours(vis_image, [approx], -1, (0, 255, 0), 4)
        if match:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            text_size, _ = cv2.getTextSize(match, font, font_scale, thickness)
            text_x = x + (w - text_size[0]) // 2
            text_y = y - 10 if y - 10 > 0 else y + h + 30
            cv2.putText(vis_image, match, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    vis_path = "detected_cards.jpg"
    cv2.imwrite(vis_path, vis_image)
    t9 = time.time()
    # read={t1-t0:.3f}s decode={t2-t1:.3f}s preprocess={t3-t2:.3f}s contours={t4-t3:.3f}s sift={t5-t4:.3f}s faiss={t7-t6:.3f}s assign={t8-t7:.3f}s draw={t9-t8:.3f}s 
    print(f"TIMING: total={t9-t0:.3f}s")
    return {"matches": card_matches}



# import asyncio
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.chrome.options import Options

# @app.get("/cardInfo")
# async def get_card_info(cardname: str):
#     """Get card info from CardMarket using Selenium."""
#     def fetch_card_info():
#         # Set up Selenium WebDriver
#         chrome_options = Options()
#         # chrome_options.add_argument("--headless")  # Run in headless mode
#         # chrome_options.add_argument("--disable-gpu")
#         # chrome_options.add_argument("--no-sandbox")
#         driver = webdriver.Chrome(options=chrome_options)

#         try:
#             # Navigate to the CardMarket search page
#             url = f"https://www.cardmarket.com/en/DragonBallSuper/Products/Search?searchString={cardname.split('_')[0]}"
#             driver.get(url)
#             card_prefix = cardname.split("-")[0].split("_")[0].upper()

#             # Wait for the table to load
#             WebDriverWait(driver, 10).until(
#                 EC.presence_of_element_located((By.CLASS_NAME, "table-body"))
#             )

#             # Find all rows in the table
#             table_body = driver.find_element(By.CLASS_NAME, "table-body")
#             rows = [row for row in table_body.find_elements(By.CLASS_NAME, "row.g-0") if "productRow" in row.get_attribute("id")]

#             # Parse the rows to find the matching card
#             for row in rows:
#                 try:
#                     # Extract the expansion (e.g., FB03, FWUP)
#                     expansion_element = row.find_element(By.CLASS_NAME, "col-icon.small")
#                     expansion = expansion_element.find_element(By.TAG_NAME, "span").text.strip()

#                     # Check if the expansion matches the card prefix
#                     if expansion == card_prefix:
#                         # Extract card details
#                         card_name = row.find_element(By.TAG_NAME, "a").text.strip()
#                         card_number = row.find_element(By.CLASS_NAME, "col-number").text.strip()
#                         availability = row.find_element(By.CLASS_NAME, "col-availability").text.strip()
#                         price = row.find_element(By.CLASS_NAME, "col-price").text.strip()
#                         link = row.find_element(By.CLASS_NAME, "col-12.col-md-8.px-2.flex-column").find_element(By.TAG_NAME, "a").get_attribute("href")
#                         # Return the matching card details
#                         return {
#                             "name": cardname,
#                             "number": card_number,
#                             "availability": availability,
#                             "price": price,
#                             "expansion": expansion,
#                             "link": link
#                         }
#                 except Exception as e:
#                     # Skip rows with missing data
#                     print(f"Error processing row: {e}")

#             # If no matching card is found
#             return {"error": "No matching card found"}

#         finally:
#             driver.quit()

#     # Run the blocking Selenium operation in a separate thread
#     return await asyncio.to_thread(fetch_card_info)


def add_new_cards_to_faiss():
    """
    Add new cards from the database to the FAISS index and update filenames.pkl.
    Only cards not already in FILENAMES are added.
    """
    descriptors_by_filename = load_descriptors()
    new_filenames = [fn for fn in descriptors_by_filename if fn not in FILENAMES]
    if not new_filenames:
        print("No new cards to add.")
        return {"added": 0, "message": "No new cards to add."}

    new_descriptors = []
    for fn in new_filenames:
        for desc in descriptors_by_filename[fn]:
            new_descriptors.append(desc)
            FILENAMES.append(fn)

    if not new_descriptors:
        print("No new descriptors to add.")
        return {"added": 0, "message": "No new descriptors to add."}

    new_descriptors = np.vstack(new_descriptors).astype(np.float32)
    FAISS_INDEX.add(new_descriptors)

    # Save updated FAISS index and filenames
    faiss.write_index(FAISS_INDEX, FAISS_FILE)
    with open(FILENAMES_FILE, "wb") as f:
        pickle.dump(FILENAMES, f)
    print(f"Added {len(new_filenames)} new cards to FAISS index and filenames.")
    return {"added": len(new_filenames), "message": f"Added {len(new_filenames)} new cards."}

@app.post("/faiss/reindex")
async def faiss_reindex():
    """API endpoint to add new cards to the FAISS index and filenames."""
    result = add_new_cards_to_faiss()
    return result

