import os
import cv2
import numpy as np
import faiss
import pickle
import argparse
from tqdm import tqdm
import os
import cv2
import numpy as np
import faiss
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

sift = cv2.SIFT_create()

def extract_features(file_path):
    """Worker function to process a single image."""
    image = cv2.imread(file_path)
    if image is None:
        return None, None
    _, descriptors = sift.detectAndCompute(image, None)
    if descriptors is not None:
        return descriptors, os.path.basename(file_path)
    return None, None

def build_faiss_from_images(cards_dir):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    file_paths = [os.path.join(cards_dir, f) for f in os.listdir(cards_dir) 
                  if f.lower().endswith(valid_extensions)]
    temp_descriptor_file = "temp_descriptors.bin"
    dimension = 128
    filename_metadata = [] 

    with open(temp_descriptor_file, "wb") as f_out:
        with ProcessPoolExecutor() as executor:
            for descriptors, filename in tqdm(executor.map(extract_features, file_paths), total=len(file_paths)):
                if descriptors is not None:
                    desc_float32 = descriptors.astype(np.float32)
                    f_out.write(desc_float32.tobytes())
                    filename_metadata.append((filename, len(desc_float32)))
    total_descriptors = sum(m[1] for m in filename_metadata)
    all_descriptors = np.memmap(temp_descriptor_file, dtype='float32', mode='r', 
                                shape=(total_descriptors, dimension))
    # 3. FAISS Setup
    nlist = 8192 #1024
    m = 32 #16
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)

    print("Training on 2M random samples...")
    idxs = np.random.choice(total_descriptors, 2_000_000, replace=False)
    train_sample = all_descriptors[idxs]
    index.train(train_sample)
    del train_sample # Clear RAM

    # 5. Add in batches
    print("Adding all descriptors to index...")
    batch_size = 500_000
    for i in range(0, total_descriptors, batch_size):
        end = min(i + batch_size, total_descriptors)
        index.add(all_descriptors[i:end])
        
    final_filenames = []
    for name, count in filename_metadata:
        final_filenames.extend([name] * count)

    return index, final_filenames

def save_faiss_index(index, filenames, faiss_file, filenames_file):
    """Save FAISS index and filenames to disk."""
    faiss.write_index(index, faiss_file)
    with open(filenames_file, "wb") as f:
        pickle.dump(filenames, f)

if __name__ == '__main__':

    STRATEGIES = ['onepiece', 'dbs', 'riftbound', 'pokemon', 'magic', 'digimon']

    parser = argparse.ArgumentParser(description='Scrape expansions with chosen strategy.')
    parser.add_argument('-s', '-strategy', '--strategy', type=str, choices=STRATEGIES, required=True,
                        help='Strategy to use for scraping expansions.')
    args = parser.parse_args()

    if os.path.exists(f"indexes/{args.strategy}/faiss_index.bin") and os.path.exists(f"indexes/{args.strategy}/filenames.pkl"):
        print(f"Indexes for strategy '{args.strategy}' already exist. Exiting.")
        exit(0)

    CARDS_DIR = f"D:\\github\\cardmarket-scraper\\card_images\\{args.strategy}"

    sift = cv2.SIFT_create()
    index, filenames = build_faiss_from_images(cards_dir=CARDS_DIR)
    os.makedirs(f"indexes/{args.strategy}", exist_ok=True)
    save_faiss_index(index, filenames, f"indexes/{args.strategy}/faiss_index.bin", f"indexes/{args.strategy}/filenames.pkl")
    print("FAISS index and filenames saved.")
