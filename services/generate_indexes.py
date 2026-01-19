import os
import cv2
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from core.config import settings


def _create_sift():
    return cv2.SIFT_create()


def _extract_features_worker(file_path):
    sift = _create_sift()
    image = cv2.imread(file_path)
    if image is None:
        return None, None
    _, descriptors = sift.detectAndCompute(image, None)
    if descriptors is not None:
        return descriptors, os.path.basename(file_path)
    return None, None


def build_faiss_from_images(cards_dir: str):
    """Build a FAISS index from all images in `cards_dir`.

    Returns (index, filenames_list).
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    file_paths = [os.path.join(cards_dir, f) for f in os.listdir(cards_dir)
                  if f.lower().endswith(valid_extensions)]
    temp_descriptor_file = "temp_descriptors.bin"
    dimension = 128
    filename_metadata = []

    with open(temp_descriptor_file, "wb") as f_out:
        with ProcessPoolExecutor() as executor:
            for descriptors, filename in tqdm(executor.map(_extract_features_worker, file_paths), total=len(file_paths)):
                if descriptors is not None:
                    desc_float32 = descriptors.astype(np.float32)
                    f_out.write(desc_float32.tobytes())
                    filename_metadata.append((filename, len(desc_float32)))

    total_descriptors = sum(m[1] for m in filename_metadata)
    if total_descriptors == 0:
        raise RuntimeError("No descriptors found in directory: %s" % cards_dir)

    all_descriptors = np.memmap(temp_descriptor_file, dtype='float32', mode='r', shape=(total_descriptors, dimension))

    # FAISS Setup (parameters preserved from original script)
    nlist = 8192
    m = 32
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)

    # Train
    print("Training index...")
    sample_count = min(2_000_000, total_descriptors)
    if sample_count < 1000:
        # fallback to smaller sample if dataset is small
        idxs = np.random.choice(total_descriptors, sample_count, replace=False)
    else:
        idxs = np.random.choice(total_descriptors, sample_count, replace=False)
    train_sample = all_descriptors[idxs]
    index.train(train_sample)
    del train_sample

    # Add descriptors in batches
    print("Adding descriptors to index...")
    batch_size = 500_000
    for i in range(0, total_descriptors, batch_size):
        end = min(i + batch_size, total_descriptors)
        index.add(all_descriptors[i:end])

    final_filenames = []
    for name, count in filename_metadata:
        final_filenames.extend([name] * count)

    # cleanup temp file
    try:
        os.remove(temp_descriptor_file)
    except Exception:
        pass

    return index, final_filenames


def save_faiss_index(index, filenames, faiss_file, filenames_file):
    os.makedirs(os.path.dirname(faiss_file), exist_ok=True)
    faiss.write_index(index, faiss_file)
    with open(filenames_file, "wb") as f:
        pickle.dump(filenames, f)


def generate_for_tcg(tcg_slug: str, force: bool = False):
    """Generate FAISS index for a configured TCG slug (settings.supported_tcgs.name).

    Returns True if created or already exists.
    """
    tcg = next((t for t in settings.supported_tcgs if t.name == tcg_slug), None)
    if not tcg:
        raise ValueError(f"TCG slug not found in config: {tcg_slug}")

    target_dir = Path("indexes") / tcg.name
    faiss_file = str(target_dir / "faiss_index.bin")
    filenames_file = str(target_dir / "filenames.pkl")

    if not force and os.path.exists(faiss_file) and os.path.exists(filenames_file):
        print(f"Indexes for '{tcg.name}' already exist. Skipping.")
        return True

    cards_dir = Path(settings.images_path) / tcg.tcg_name
    if not cards_dir.exists():
        raise FileNotFoundError(f"Cards directory not found: {cards_dir}")

    index, filenames = build_faiss_from_images(str(cards_dir))
    save_faiss_index(index, filenames, faiss_file, filenames_file)
    print(f"Saved index for {tcg.name}")
    return True


def generate_all(force: bool = False, tcg_list=None):
    """Generate indexes for all configured tcgs or for an optional list of slugs."""
    slugs = tcg_list or [t.name for t in settings.supported_tcgs]
    for slug in slugs:
        try:
            generate_for_tcg(slug, force=force)
        except Exception as e:
            print(f"Failed to generate for {slug}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate FAISS indexes for configured TCGs")
    parser.add_argument("--strategy", "-s", help="TCG slug to build (e.g., riftbound)")
    parser.add_argument("--force", action="store_true", help="Rebuild even if index exists")
    args = parser.parse_args()

    if args.strategy:
        generate_for_tcg(args.strategy, force=args.force)
    else:
        generate_all(force=args.force)
