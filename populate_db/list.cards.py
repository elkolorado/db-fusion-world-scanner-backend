import sqlite3

def list_cards(db_path):
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query to fetch all cards
        cursor.execute("SELECT * FROM cards")
        cards = cursor.fetchall()

        # Print the cards
        for card in cards:
            print(card)


            # Check if the card exists in the cards table
        card_id = "FP-010.webp"  # Assuming the first column is the ID
        cursor.execute("SELECT id FROM cards WHERE filename = ?", (card_id,))
        result = cursor.fetchone()
        if result:
            print(f"Card with ID {card_id} exists.")
        else:
            print(f"Card with ID {card_id} does not exist.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    db_path = "cards.db"  # Update with the correct path if needed
    list_cards(db_path)

    # clear user_cards table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user_cards")
    conn.commit()