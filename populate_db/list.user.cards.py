import sqlite3

def list_user_cards(user_id):
    """
    Fetch and list all cards associated with a specific user from the database.

    :param user_id: ID of the user whose cards are to be listed
    :return: List of cards
    """
    try:
        # Connect to the database
        conn = sqlite3.connect('cards.db')
        cursor = conn.cursor()

        # Query to fetch user cards
        query = """
        SELECT user_cards.card_id, user_cards.quantity
        FROM user_cards
        WHERE user_cards.username = ?
        """
        cursor.execute(query, (user_id,))
        cards = cursor.fetchall()

        # Print the cards
        if cards:
            print(f"Cards for User ID {user_id}:")
            for card in cards:
                print(card)
        else:
            print(f"No cards found for User ID {user_id}.")

        return cards

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        if conn:
            conn.close()

# Example usage
if __name__ == "__main__":
    list_user_cards("123")