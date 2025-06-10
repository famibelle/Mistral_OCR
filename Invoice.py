import base64
import os
import sqlite3
import streamlit as st
from PIL import Image
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk

import json


def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def check_and_update_database(data, image_path):
    """Check if the data exists in the database and update or insert accordingly."""
    conn = sqlite3.connect('factures.db')
    cursor = conn.cursor()

    # Création de la table si elle n'existe pas
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Factures (
            facture_id INTEGER PRIMARY KEY AUTOINCREMENT,
            numero_facture TEXT UNIQUE,                -- Numéro de facture
            date_emission TEXT,                        -- Date d'émission
            vendeur_nom TEXT,                          -- Nom ou raison sociale du vendeur
            vendeur_adresse TEXT,                      -- Adresse complète du vendeur
            vendeur_siret TEXT,                        -- Numéro SIRET ou SIREN du vendeur
            vendeur_tva TEXT,                          -- Numéro de TVA intracommunautaire du vendeur
            client_nom TEXT,                           -- Nom ou raison sociale du client
            client_adresse TEXT,                       -- Adresse complète du client
            description TEXT,                          -- Description des biens ou services
            date_vente TEXT,                           -- Date de la vente ou prestation (si différente)
            prix_unitaire_ht REAL,                     -- Prix unitaire HT
            quantite INTEGER,                          -- Quantité
            taux_tva TEXT,                             -- Taux de TVA applicable
            montant_ht REAL,                           -- Montant total HT
            montant_tva REAL,                          -- Montant TVA
            montant_ttc REAL,                          -- Montant total TTC
            conditions_paiement TEXT,                  -- Conditions de paiement
            mentions_legales TEXT,                     -- Mentions légales spécifiques
            image_path TEXT                            -- Chemin de l'image de la facture
        )
    ''')

    # Insertion des données extraites
    cursor.execute('''
        INSERT OR IGNORE INTO Factures (
            numero_facture, date_emission, vendeur_nom, vendeur_adresse, vendeur_siret, vendeur_tva,
            client_nom, client_adresse, description, date_vente, prix_unitaire_ht, quantite,
            taux_tva, montant_ht, montant_tva, montant_ttc, conditions_paiement, mentions_legales, image_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('numero_facture'),
        data.get('date_emission'),
        data.get('vendeur_nom'),
        data.get('vendeur_adresse'),
        data.get('vendeur_siret'),
        data.get('vendeur_tva'),
        data.get('client_nom'),
        data.get('client_adresse'),
        data.get('description'),
        data.get('date_vente'),
        data.get('prix_unitaire_ht'),
        data.get('quantite'),
        data.get('taux_tva'),
        data.get('montant_ht'),
        data.get('montant_tva'),
        data.get('montant_ttc'),
        data.get('conditions_paiement'),
        data.get('mentions_legales'),
        image_path
    ))

    conn.commit()
    conn.close()

def main():
    st.title("Système de Gestion de Factures avec Mistral OCR")

    # Sidebar for API key input
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Entrez votre clé API Mistral", type="password")

    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png"])
    if uploaded_file is not None:
        # Save the uploaded image
        image_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Image téléchargée', use_container_width=True)

        # Encode the image to base64
        base64_image = encode_image(image_path)

        if api_key:
            # Use Mistral API for OCR
            client = Mistral(api_key=api_key)
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                },
                include_image_base64=True,
                #document_annotation_format = { "type": "json_object" }
            )
            image_ocr_markdown = ocr_response.pages[0].markdown

            # Display OCR results
            #st.subheader("Informations extraites")
            #st.write(image_ocr_markdown)


            chat_response = client.chat.complete(
                model="pixtral-12b-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            ImageURLChunk(image_url=f"data:image/jpeg;base64,{base64_image}"),
                            TextChunk(text=f"""
Voici le résultat de l'OCR :

{image_ocr_markdown}

Peux-tu extraire les informations de la facture? L'output doit être au format JSON **à plat** (pas de listes ni d'objets imbriqués).

Les champs à extraire sont les suivants:
- numero_facture
- date_emission
- vendeur_nom
- vendeur_adresse
- vendeur_siret
- vendeur_tva
- client_nom
- client_adresse
- description : concatène toutes les lignes d'articles en une seule chaîne de texte lisible (exemple : "Produit A x2 - 10€ HT, Produit B x1 - 5€ HT")
- date_vente
- prix_unitaire_ht
- quantite
- taux_tva
- montant_ht
- montant_tva
- montant_ttc
- conditions_paiement
- mentions_legales

**Le JSON doit être à plat, chaque champ doit être une clé de premier niveau.**
Assure-toi que le JSON est valide et que les champs sont bien formatés.
"""),
                        ],
                    },
                ],
                response_format= {"type": "json_object"},
                temperature=0
            )
            response_dict = json.loads(chat_response.choices[0].message.content)
            json_string = json.dumps(response_dict, indent=4)
            st.subheader("Informations extraites")
            st.json(json_string)

            # Update the database with OCR results
            check_and_update_database(response_dict, image_path)
            st.success("✅ Transmis")
        else:
            st.error("Veuillez entrer une clé API Mistral valide.")

    # Option pour afficher les factures
    st.subheader("Afficher les factures enregistrées")
    conn = sqlite3.connect('factures.db')
    df = None
    try:
        df = None
        import pandas as pd
        df = pd.read_sql_query("SELECT * FROM Factures", conn)
    except Exception as e:
        st.error(f"Erreur lors de la lecture de la base : {e}")
    finally:
        conn.close()
    if df is not None and not df.empty:
        st.subheader("Factures enregistrées")
        # Surligner la dernière facture ajoutée si possible
        if 'numero_facture' in df.columns and 'response_dict' in locals():
            last_num = response_dict.get('numero_facture')
            def highlight_last(row):
                style = 'font-weight: bold' if row['numero_facture'] == last_num else ''
                return [style]*len(row)
            st.dataframe(df.style.apply(highlight_last, axis=1))
        else:
            st.dataframe(df)

    else:
        st.info("Aucune facture enregistrée.")

if __name__ == "__main__":
    main()
