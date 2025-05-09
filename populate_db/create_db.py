import sqlite3

def create_database(db_path="cards.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()




    conn.commit()
    conn.close()

# Create the database
create_database()