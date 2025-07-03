import subprocess
import os

def run_initialization():
    """Run initialization scripts."""
    try:
        # Check if faiss_index.bin exists
        if not os.path.exists("faiss_index.bin"):

            #delete cards.db if it exists
            if os.path.exists("cards.db"):
                os.remove("cards.db")

            # Run db.py
            print("Running db.py...")
            subprocess.run(["python", "populate_db/create_users_table.py"], check=True)

            # Run lightglue/db.py
            print("Running lightglue/db.py...")
            subprocess.run(["python", "populate_db/db.py"], check=True)
        else:
            print("faiss_index.bin already exists. Skipping database initialization.")

        # Run uvicorn
        print("Starting uvicorn server...")
        subprocess.run(["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"], check=True)

        print("Initialization completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Initialization failed: {e}")
        exit(1)

if __name__ == "__main__":
    run_initialization()