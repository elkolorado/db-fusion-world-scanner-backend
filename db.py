import sqlite3

def create_database(db_path="cards.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the cards table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE NOT NULL
    );
    """)

    # Create the keypoints table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS keypoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        descriptor BLOB NOT NULL,
        FOREIGN KEY (card_id) REFERENCES cards(id)
    );
    """)

    conn.commit()
    conn.close()

# Create the database
create_database()