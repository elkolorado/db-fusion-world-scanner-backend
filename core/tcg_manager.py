import faiss
import pickle
import os
from core.config import settings

class TCGManager:
    def __init__(self):
        self.indexes = {}
        self.filenames = {}
        self.tcg_data = settings.config.supported_tcgs 

    def load_indexes(self):
        """Loads FAISS indexes and filenames for the current environment."""
        for tcg in self.tcg_data:
            # Use tcg.name (e.g., 'riftbound', 'dbs') for folder paths
            idx_path = f"indexes/{tcg.name}/faiss_index.bin"
            pkl_path = f"indexes/{tcg.name}/filenames.pkl"
            
            if not os.path.exists(idx_path) or not os.path.exists(pkl_path):
                print(f"⚠️ Warning: Missing files for {tcg.name}. Skipping...")
                continue

            index = faiss.read_index(idx_path)
            index.nprobe = tcg.nprobes
            
            with open(pkl_path, "rb") as f:
                filenames = pickle.load(f)
            
            self.indexes[tcg.name] = index
            self.filenames[tcg.name] = filenames
            
            print(f"✅ Loaded {tcg.tcg_name}: {len(filenames)} cards (nprobe={tcg.nprobes})")

# Create the singleton instance
tcg_manager = TCGManager()