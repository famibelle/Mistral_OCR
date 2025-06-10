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

    # Create the Factures table if it doesn't exist
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

    # Insert data into the Factures table
    cursor.execute('''
        INSERT INTO Factures (
            etablissement, site_web, type_client, bar_code, bar_numero, duplicata, date, heure,
            ticket_reference, ticket_description, ticket_quantite, ticket_prix, total_montant_ht,
            total_montant_ttc, tva_taux, tva_montant, siret, ape, tva_etablissement, adresse,
            logiciel, fiche, image_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('etablissement'), data.get('site_web'), data.get('type'), data.get('bar', {}).get('code'),
        data.get('bar', {}).get('numero'), data.get('duplicata'), data.get('date'), data.get('heure'),
        data.get('ticket', {}).get('reference'), data.get('ticket', {}).get('description'),
        data.get('ticket', {}).get('quantite'), data.get('ticket', {}).get('prix'),
        data.get('total', {}).get('montant_ht'), data.get('total', {}).get('montant_ttc'),
        data.get('total', {}).get('tva', {}).get('taux'), data.get('total', {}).get('tva', {}).get('montant'),
        data.get('siret'), data.get('ape'), data.get('tva_etablissement'), data.get('adresse'),
        data.get('logiciel'), data.get('fiche'), image_path
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
            st.subheader("Informations extraites")
            st.write(image_ocr_markdown)


            chat_response = client.chat.complete(
                model="pixtral-12b-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            ImageURLChunk(image_url=f"data:image/jpeg;base64,{base64_image}"),
                            TextChunk(text=f"Voici le résultat de l'OCR :\n\n{image_ocr_markdown}\n\nPeux-tu extraire les informations de la facture ? L'output doit être au format JSON."),
                        ],
                    },
                ],
                response_format= {"type": "json_object"},
                temperature=0
            )
            response_dict = json.loads(chat_response.choices[0].message.content)
            json_string = json.dumps(response_dict, indent=4)
            st.json(json_string)

            # Update the database with OCR results
            check_and_update_database(response_dict, image_path)
            st.success("✅ Transmis")
        else:
            st.error("Veuillez entrer une clé API Mistral valide.")

if __name__ == "__main__":
    main()
