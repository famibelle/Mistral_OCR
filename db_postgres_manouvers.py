import os
import sys
import sqlalchemy
from sqlalchemy import create_engine, text

def show_readable_db():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("Veuillez définir la variable d'environnement DATABASE_URL.")
        return

    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM Factures"))
        rows = result.fetchall()
        if not rows:
            print("La table Factures est vide.")
        else:
            headers = result.keys()
            print(" | ".join(headers))
            print("-" * (len(headers) * 15))
            for row in rows:
                print(" | ".join(str(row[col]) for col in headers))

def alter_db_constraints():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("Veuillez définir la variable d'environnement DATABASE_URL.")
        return

    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        # 1. Supprimer l'ancienne contrainte UNIQUE (remplace le nom si besoin)
        conn.execute(text('ALTER TABLE Factures DROP CONSTRAINT IF EXISTS factures_date_vente_heure_montant_ht_montant_ttc_key'))
        # 2. Ajouter la nouvelle contrainte UNIQUE
        conn.execute(text('ALTER TABLE Factures ADD CONSTRAINT factures_unique_date_vente_heure_montant_ht_numero_facture UNIQUE(date_vente, heure, montant_ht, numero_facture)'))
    print("Contraintes modifiées avec succès.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python db_postgres_manouvers.py [show|alter]")
    elif sys.argv[1] == "show":
        show_readable_db()
    elif sys.argv[1] == "alter":
        alter_db_constraints()
    else:
        print("Argument inconnu. Utilisez 'show' pour afficher la base ou 'alter' pour modifier les contraintes.")