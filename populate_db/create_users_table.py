import sqlite3
import bcrypt
def add_user_tables(db_path="cards.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()


    #clear the database
    cursor.execute("DROP TABLE IF EXISTS user_cards")
    cursor.execute("DROP TABLE IF EXISTS users")
    cursor.execute("DROP TABLE IF EXISTS cards")
    cursor.execute("DROP TABLE IF EXISTS keypoints")

    # Create the users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
    );
    """)

    # Create the user_cards table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        card_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL DEFAULT 1,
        FOREIGN KEY (username) REFERENCES users(username),
        FOREIGN KEY (card_id) REFERENCES cards(id)
    );
    """)


    # # Add a dummy user
    cursor.execute("""
        INSERT OR IGNORE INTO users (username, password)
        VALUES (?, ?)
    """, ("123", bcrypt.hashpw("123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')))


    conn.commit()
    conn.close()

# Add the tables to the database
add_user_tables()

def create_required_tables(db_path="cards.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the cards table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE NOT NULL,
        name TEXT,
        "set" TEXT
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

# Create the required tables
create_required_tables()