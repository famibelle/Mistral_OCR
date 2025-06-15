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
                print(" | ".join(str(value) for value in row))

def alter_db_constraints():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("Veuillez définir la variable d'environnement DATABASE_URL.")
        return

    engine = create_engine(DATABASE_URL)
    try:
        with engine.connect() as conn:
            # 1. Supprimer l'ancienne contrainte UNIQUE (remplace le nom si besoin)
            conn.execute(text('ALTER TABLE Factures DROP CONSTRAINT IF EXISTS factures_date_vente_heure_montant_ht_montant_ttc_key'))
            # 2. Ajouter la nouvelle contrainte UNIQUE
            conn.execute(text('ALTER TABLE Factures ADD CONSTRAINT factures_unique_date_vente_heure_montant_ht_numero_facture UNIQUE(date_vente, heure, montant_ht, numero_facture)'))
        print("Contraintes modifiées avec succès.")
    except Exception as e:
        print(f"Erreur lors de la modification des contraintes : {e}")

def show_table_structure():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("Veuillez définir la variable d'environnement DATABASE_URL.")
        return

    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        print("Structure de la table Factures :")
        result = conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'factures'
            ORDER BY ordinal_position
        """))
        rows = result.fetchall()
        if not rows:
            print("La table Factures n'existe pas.")
        else:
            print("Nom de colonne | Type | Nullable")
            print("-" * 40)
            for row in rows:
                print(f"{row[0]} | {row[1]} | {row[2]}")

def drop_factures_table():
    DATABASE_URL = os.getenv("DATABASE_URL")
    print(f"Connexion à la base : {DATABASE_URL}")
    if not DATABASE_URL:
        print("Veuillez définir la variable d'environnement DATABASE_URL.")
        return

    engine = create_engine(DATABASE_URL)
    try:
        with engine.begin() as conn:
            conn.execute(text('DROP TABLE IF EXISTS Factures CASCADE'))
        print("Table Factures supprimée avec succès.")
    except Exception as e:
        print(f"Erreur lors de la suppression de la table : {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python db_postgres_manouvers.py [show|alter|structure|drop]")
    elif sys.argv[1] == "show":
        show_readable_db()
    elif sys.argv[1] == "alter":
        alter_db_constraints()
    elif sys.argv[1] == "structure":
        show_table_structure()
    elif sys.argv[1] == "drop":
        drop_factures_table()
    else:
        print("Argument inconnu. Utilisez 'show', 'alter', 'structure' ou 'drop'.")