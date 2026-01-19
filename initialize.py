import subprocess
from services import generate_indexes


def run_initialization():
    """Run initialization scripts (generate indexes), then start the app."""
    try:
        # Generate indexes for configured TCGs if missing
        print("Generating FAISS indexes (if missing)...")
        generate_indexes.generate_all(force=False)

        # Start the app
        subprocess.run(["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"], check=True)

        print("Initialization completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Initialization failed: {e}")
        exit(1)
    except Exception as e:
        print(f"Initialization error: {e}")
        exit(1)


if __name__ == "__main__":
    run_initialization()