import base64
import os
import sqlite3
import streamlit as st
from PIL import Image

# Mockup de la classe Mistral pour l'exemple
class Mistral:
    def __init__(self, api_key):
        self.api_key = api_key

    def ocr_process(self, model, document):
        # Simulation de la réponse de l'API Mistral
        return {
            "etablissement": "Le Chalet des Iles",
            "site_web": "www.lechaletdesiles.net",
            "type": "Client de passage",
            "bar": {
                "code": "1/137726-Matthieu",
                "numero": "A724276.86766"
            },
            "duplicata": "C724276.11945",
            "date": "08/06/2025",
            "heure": "18:13:33",
            "ticket": {
                "reference": "R724276.86081",
                "description": "Repas complet",
                "quantite": 1,
                "prix": 12.00
            },
            "total": {
                "montant_ht": 10.00,
                "montant_ttc": 12.00,
                "tva": {
                    "taux": "20%",
                    "montant": 2.00
                }
            },
            "siret": "78463604500014",
            "ape": "5610A",
            "tva_etablissement": "FR29784636045",
            "adresse": "75016 Paris, France",
            "logiciel": "Lightspeed (K) 25.20.1.29733",
            "fiche": "F724276.7702",
            "image_path": "path_to_image.jpg"
        }

def encode_image(image_path):
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def check_and_update_database(data, image_path):
    """Check if the data exists in the database and update or insert accordingly."""
    conn = sqlite3.connect('factures.db')
    cursor = conn.cursor()

    # Création de la table Factures avec tous les champs nécessaires
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Factures (
            facture_id INTEGER PRIMARY KEY AUTOINCREMENT,
            etablissement TEXT,
            site_web TEXT,
            type_client TEXT,
            bar_code TEXT,
            bar_numero TEXT,
            duplicata TEXT,
            date TEXT,
            heure TEXT,
            ticket_reference TEXT,
            ticket_description TEXT,
            ticket_quantite INTEGER,
            ticket_prix REAL,
            total_montant_ht REAL,
            total_montant_ttc REAL,
            tva_taux TEXT,
            tva_montant REAL,
            siret TEXT,
            ape TEXT,
            tva_etablissement TEXT,
            adresse TEXT,
            logiciel TEXT,
            fiche TEXT,
            image_path TEXT
        )
    ''')

    # Insertion des données dans la table Factures
    cursor.execute('''
        INSERT INTO Factures (
            etablissement, site_web, type_client, bar_code, bar_numero, duplicata, date, heure,
            ticket_reference, ticket_description, ticket_quantite, ticket_prix, total_montant_ht,
            total_montant_ttc, tva_taux, tva_montant, siret, ape, tva_etablissement, adresse,
            logiciel, fiche, image_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['etablissement'], data['site_web'], data['type'], data['bar']['code'], data['bar']['numero'],
        data['duplicata'], data['date'], data['heure'], data['ticket']['reference'], data['ticket']['description'],
        data['ticket']['quantite'], data['ticket']['prix'], data['total']['montant_ht'], data['total']['montant_ttc'],
        data['total']['tva']['taux'], data['total']['tva']['montant'], data['siret'], data['ape'],
        data['tva_etablissement'], data['adresse'], data['logiciel'], data['fiche'], image_path
    ))

    conn.commit()
    conn.close()

def main():
    st.title("Système de Gestion de Factures avec Mistral OCR")

    # Ajout d'une barre latérale pour la saisie de la clé API
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Entrez votre clé API Mistral", type="password")

    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png"])
    if uploaded_file is not None:
        # Sauvegarder l'image téléchargée
        image_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Afficher l'image
        image = Image.open(uploaded_file)
        st.image(image, caption='Image téléchargée', use_column_width=True)

        # Encoder l'image en base64
        base64_image = encode_image(image_path)

        # Utiliser l'API Mistral pour l'OCR
        if api_key:
            client = Mistral(api_key=api_key)
            ocr_response = client.ocr_process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            )

            # Afficher les résultats de l'OCR
            st.subheader("Informations extraites")
            st.write(ocr_response)

            # Mettre à jour la base de données avec les résultats de l'OCR
            check_and_update_database(ocr_response, image_path)
            st.success("✅ Transmis")
        else:
            st.error("Veuillez entrer une clé API Mistral valide.")

if __name__ == "__main__":
    main()
