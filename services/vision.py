import time
import io
from PIL import Image, ImageOps
import cv2
import numpy as np

# SIFT detector
sift = cv2.SIFT_create(nfeatures=8192)


def extract_keypoints_from_bytes(image_bytes):
    """Returns (keypoints, descriptors) for given image bytes."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return [], None

    img_pil = Image.open(io.BytesIO(image_bytes))
    img_pil = ImageOps.exif_transpose(img_pil)
    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    max_dim = 1000
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def match_card(image_bytes, faiss_index, filenames):
    """Synchronous matching function that uses FAISS index and filenames list.
    Returns best_match filename (or None).
    """
    start_time = time.time()
    _, query_descriptors = extract_keypoints_from_bytes(image_bytes)
    if query_descriptors is None or len(query_descriptors) == 0:
        return None

    query_descriptors = np.array(query_descriptors, dtype=np.float32)

    k = 2
    distances, indices = faiss_index.search(query_descriptors, k)

    # Lowe ratio test
    mask = distances[:, 0] < 0.75 * distances[:, 1]
    good_matches_indices = indices[mask, 0]

    if len(good_matches_indices) == 0:
        return None

    match_counts = {}
    for match_idx in good_matches_indices:
        filename = filenames[match_idx]
        match_counts[filename] = match_counts.get(filename, 0) + 1

    best_match = max(match_counts, key=match_counts.get, default=None)
    return best_match
