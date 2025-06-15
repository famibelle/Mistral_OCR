import sqlite3

def show_readable_db():
    conn = sqlite3.connect('factures.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    rows = cursor.execute("SELECT * FROM Factures").fetchall()
    if not rows:
        print("La table Factures est vide.")
    else:
        # Afficher les en-têtes
        headers = rows[0].keys()
        print(" | ".join(headers))
        print("-" * (len(headers) * 15))

        # Afficher chaque ligne sous forme lisible
        for row in rows:
            print(" | ".join(str(row[col]) for col in headers))

    conn.close()

if __name__ == "__main__":
    show_readable_db()