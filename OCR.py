import base64
import requests

import os
import streamlit as st
from mistralai import Mistral
import json

# Configuration de la page Streamlit
st.title("Mistral OCR Interface")
st.write("Téléchargez une image pour extraire le texte avec Mistral OCR.")

def encode_image(image_file):
    """Encode the image to base64."""
    try:
        return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None

# Entrée pour la clé API
api_key = st.text_input("Entrez votre clé API Mistral", type="password")

# Téléchargement de l'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

# Getting the base64 string

if uploaded_file is not None:
    # Afficher l'image téléchargée
    st.image(uploaded_file, caption='Image téléchargée', use_container_width=True)
    base64_image = encode_image(uploaded_file)

    # Bouton pour envoyer l'image à Mistral OCR
    if st.button("Extraire le texte"):
        if api_key:
            try:
                # Initialiser le client Mistral
                client = Mistral(api_key=api_key)

                # Envoyer l'image à Mistral OCR
                ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}" 
                    },
                    include_image_base64=True
                )

                # Afficher le texte extrait
                st.subheader("Texte extrait")

                st.markdown(ocr_response.pages[0].markdown)
                print(ocr_response)

            except Exception as e:
                st.error(f"Une erreur s'est produite: {e}")
        else:
            st.error("Veuillez entrer une clé API valide.")
