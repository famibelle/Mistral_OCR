import sqlite3

conn = sqlite3.connect('factures.db')
cursor = conn.cursor()

for row in cursor.execute("SELECT * FROM Factures"):
    print(row)

conn.close()