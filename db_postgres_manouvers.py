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

def enable_fuzzystrmatch():
    """
    Vérifie si l'extension fuzzystrmatch est active ; si non, l'active.
    Affiche explicitement toute erreur rencontrée.
    """
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("Veuillez définir la variable d'environnement DATABASE_URL.")
        return

    engine = create_engine(DATABASE_URL)
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT extname
                  FROM pg_extension
                 WHERE extname = 'fuzzystrmatch'
            """)).fetchone()
            if row:
                print("L'extension fuzzystrmatch est déjà activée.")
            else:
                try:
                    conn.execute(text("CREATE EXTENSION fuzzystrmatch;"))
                    print("Extension fuzzystrmatch activée avec succès.")
                except Exception as e:
                    print(f"Erreur lors de la création de l'extension fuzzystrmatch : {e}")
    except Exception as e:
        print(f"Erreur lors de la vérification de fuzzystrmatch : {e}")

def check_fuzzystrmatch():
    """
    Vérifie si l'extension fuzzystrmatch est activée.
    """
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("Veuillez définir la variable d'environnement DATABASE_URL.")
        return

    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT extname, extversion
            FROM pg_extension
            WHERE extname = 'fuzzystrmatch'
        """))
        row = result.fetchone()
        if row:
            print(f"fuzzystrmatch est activée (version {row[1]}).")
        else:
            print("fuzzystrmatch n'est pas activée.")

def create_factures_table():
    """
    Crée la table Factures si elle n'existe pas déjà.
    """
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("Veuillez définir la variable d'environnement DATABASE_URL.")
        return

    engine = create_engine(DATABASE_URL)
    try:
        with engine.connect() as conn:
            # Vérifie si la table existe déjà (en minuscules)
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_name = 'factures'
                )
            """))
            exists = result.scalar()
            if exists:
                print("La table Factures existe déjà.")
                return

        # Si la table n'existe pas, on la crée
        with engine.begin() as conn:
            conn.execute(text('''
                CREATE TABLE Factures (
                    facture_id SERIAL PRIMARY KEY,
                    numero_facture TEXT,
                    date_emission DATE,
                    vendeur_nom TEXT,
                    vendeur_adresse TEXT,
                    vendeur_siret TEXT,
                    vendeur_tva TEXT,
                    client_nom TEXT,
                    client_adresse TEXT,
                    description TEXT,
                    date_vente DATE,
                    heure TIME,
                    prix_unitaire_ht NUMERIC(12, 2),
                    quantite INTEGER,
                    taux_tva TEXT,
                    montant_ht NUMERIC(12, 2),
                    montant_tva NUMERIC(12, 2),
                    montant_ttc NUMERIC(12, 2),
                    conditions_paiement TEXT,
                    mentions_legales TEXT,
                    image_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date_vente, heure, montant_ht, numero_facture)
                )
            '''))
        print("Table Factures créée avec succès.")
    except Exception as e:
        print(f"Erreur lors de la création de la table Factures : {e}")

# Ajoute l'option CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python db_postgres_manouvers.py [show|alter|structure|drop|fuzzystrmatch|create]")
    elif sys.argv[1] == "show":
        show_readable_db()
    elif sys.argv[1] == "alter":
        alter_db_constraints()
    elif sys.argv[1] == "structure":
        show_table_structure()
    elif sys.argv[1] == "drop":
        drop_factures_table()
    elif sys.argv[1] == "fuzzystrmatch":
        enable_fuzzystrmatch()
    elif sys.argv[1] == "check_fuzzystrmatch":
        check_fuzzystrmatch()
    elif sys.argv[1] == "create":
        create_factures_table()
    else:
        print("Argument inconnu. Utilisez 'show', 'alter', 'structure', 'drop', 'fuzzystrmatch' ou 'create'.")