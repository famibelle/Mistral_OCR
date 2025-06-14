import base64
import os
import sqlite3
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from mistralai import Mistral, ImageURLChunk, TextChunk
import json
from twilio.rest import Client
import uvicorn
import requests
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime
import dotenv
import time

# Charger les variables d'environnement à partir du fichier .env
dotenv.load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de Twilio
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
USER_PHONE_NUMBER = os.getenv('USER_PHONE_NUMBER')

# Configuration de Mistral
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

# Initialiser les clients
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()

# Modèles Pydantic
class Facture(BaseModel):
    numero_facture: Optional[str] = None
    date_emission: Optional[str] = None
    vendeur_nom: Optional[str] = None
    vendeur_adresse: Optional[str] = None
    vendeur_siret: Optional[str] = None
    vendeur_tva: Optional[str] = None
    client_nom: Optional[str] = None
    client_adresse: Optional[str] = None
    description: Optional[str] = None
    date_vente: Optional[str] = None
    prix_unitaire_ht: Optional[float] = None
    quantite: Optional[int] = None
    taux_tva: Optional[str] = None
    montant_ht: Optional[float] = None
    montant_tva: Optional[float] = None
    montant_ttc: Optional[float] = None
    conditions_paiement: Optional[str] = None
    mentions_legales: Optional[str] = None
    image_path: Optional[str] = None
    created_at: str = datetime.now().isoformat()

def encode_image(image_path: str) -> Optional[str]:
    """Encode l'image en base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logger.error(f"Fichier non trouvé: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Erreur lors de l'encodage de l'image: {e}")
        return None

def save_image(image_data: bytes, filename: str) -> str:
    """Sauvegarde l'image localement."""
    try:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        image_path = os.path.join(upload_dir, filename)
        with open(image_path, "wb") as f:
            f.write(image_data)
        return image_path
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de l'image: {e}")
        raise

def check_and_update_database(data: dict) -> list:
    """
    Vérifie si la facture existe dans la base, insère ou met à jour, et retourne la liste des champs ajoutés ou mis à jour.
    """
    updated_fields = []
    try:
        conn = sqlite3.connect('factures.db')
        cursor = conn.cursor()

        # Création de la table si elle n'existe pas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Factures (
                facture_id INTEGER PRIMARY KEY AUTOINCREMENT,
                numero_facture TEXT UNIQUE,
                date_emission TEXT,
                vendeur_nom TEXT,
                vendeur_adresse TEXT,
                vendeur_siret TEXT,
                vendeur_tva TEXT,
                client_nom TEXT,
                client_adresse TEXT,
                description TEXT,
                date_vente TEXT,
                prix_unitaire_ht REAL,
                quantite INTEGER,
                taux_tva TEXT,
                montant_ht REAL,
                montant_tva REAL,
                montant_ttc REAL,
                conditions_paiement TEXT,
                mentions_legales TEXT,
                image_path TEXT,
                created_at TEXT
            )
        ''')

        # Vérifier si la facture existe déjà
        cursor.execute(
            "SELECT * FROM Factures WHERE numero_facture = ?",
            (data.get('numero_facture'),)
        )
        existing = cursor.fetchone()
        columns = [desc[0] for desc in cursor.description]

        if not existing:
            # Nouvelle facture : insertion
            cursor.execute('''
                INSERT INTO Factures (
                    numero_facture, date_emission, vendeur_nom, vendeur_adresse, vendeur_siret, vendeur_tva,
                    client_nom, client_adresse, description, date_vente, prix_unitaire_ht, quantite,
                    taux_tva, montant_ht, montant_tva, montant_ttc, conditions_paiement, mentions_legales,
                    image_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                data.get('image_path'),
                data.get('created_at', datetime.now().isoformat())
            ))
            logger.info("nouvelles factures enregistrées")
        else:
            # Facture existante : mise à jour et log des champs modifiés
            for idx, col in enumerate(columns):
                if col == "facture_id" or col == "numero_facture":
                    continue
                new_value = data.get(col)
                old_value = existing[idx]
                # Champ ajouté ou modifié
                if (old_value is None or old_value == "") and new_value not in (None, ""):
                    updated_fields.append(col)
                elif new_value not in (None, "") and str(new_value) != str(old_value):
                    updated_fields.append(col)
            # Mise à jour de la facture
            cursor.execute('''
                UPDATE Factures SET
                    date_emission=?,
                    vendeur_nom=?,
                    vendeur_adresse=?,
                    vendeur_siret=?,
                    vendeur_tva=?,
                    client_nom=?,
                    client_adresse=?,
                    description=?,
                    date_vente=?,
                    prix_unitaire_ht=?,
                    quantite=?,
                    taux_tva=?,
                    montant_ht=?,
                    montant_tva=?,
                    montant_ttc=?,
                    conditions_paiement=?,
                    mentions_legales=?,
                    image_path=?,
                    created_at=?
                WHERE numero_facture=?
            ''', (
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
                data.get('image_path'),
                data.get('created_at', datetime.now().isoformat()),
                data.get('numero_facture')
            ))
            if updated_fields:
                logger.info(f"facture existante, champs ajoutés ou mis à jour : {', '.join(updated_fields)}")
            else:
                logger.info("facture existante, aucune donnée rajoutée ou modifiée")
        conn.commit()
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la base de données: {e}")
        raise
    finally:
        conn.close()
    return updated_fields

def process_image(image_path: str) -> dict:
    """Traite une image de facture et extrait les informations."""
    try:
        base64_image = encode_image(image_path)
        if not base64_image:
            raise HTTPException(status_code=400, detail="Impossible d'encoder l'image")

        # Utiliser Mistral API pour l'OCR
        ocr_response = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            },
            include_image_base64=True,
        )
        image_ocr_markdown = ocr_response.pages[0].markdown

        # Extraire les informations de la facture
        chat_response = mistral_client.chat.complete(
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
- description : concatène toutes les lignes d'articles en une seule chaîne de texte lisible (exemple: "Produit A x2 - 10€ HT, Produit B x1 - 5€ HT")
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
            response_format={"type": "json_object"},
            temperature=0
        )

        response_dict = json.loads(chat_response.choices[0].message.content)
        response_dict['image_path'] = image_path
        return response_dict
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'image: {e}")
        raise

def send_whatsapp_message(to: str, body: str) -> str:
    """Envoie un message WhatsApp via Twilio."""
    try:
        message = twilio_client.messages.create(
            body=body,
            from_=f'whatsapp:{TWILIO_PHONE_NUMBER}',
            to=to  # Ici, on utilise le numéro de l'expéditeur
        )
        return message.sid
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi du message WhatsApp: {e}")
        raise

def process_incoming_message(data: dict, background_tasks: BackgroundTasks) -> None:
    """Traite les messages entrants de Twilio en arrière-plan."""
    try:
        logger.info(f"Message reçu de: {data.get('From')}")
        logger.info(f"Clés reçues dans le message : {list(data.keys())}")
        body = data.get('Body', '').strip()

        # Recherche robuste de la clé MediaUrl0 (insensible à la casse)
        media_url_key = next((k for k in data.keys() if k.lower() == 'mediaurl0'), None)
        if not media_url_key or not data.get(media_url_key):
            logger.info("Aucune image trouvée dans le message, aucune action.")
            return

        # Télécharger l'image
        media_url = data[media_url_key]
        response = requests.get(media_url)
        response.raise_for_status()

        # Sauvegarder l'image localement
        filename = f"facture_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        image_path = save_image(response.content, filename)
        logger.info(f"Image sauvegardée: {image_path}")

        # Traiter l'image en arrière-plan
        background_tasks.add_task(process_and_respond, data['From'], image_path)
    except Exception as e:
        logger.error(f"Erreur lors du traitement du message entrant: {e}")


def process_and_respond(phone_number: str, image_path: str) -> None:
    """Traite l'image et envoie une réponse à l'utilisateur."""
    try:
        # 1. Réponse immédiate à la réception du ticket avec emoji d'attente
        send_whatsapp_message(phone_number, "Merci pour votre envoi ! Je m'en occupe... ⏳")

        # 2. Traitement du ticket
        facture_data = process_image(image_path)
        logger.info(f"Données extraites: {facture_data}")
        updated_fields = check_and_update_database(facture_data)
        logger.info("Base de données mise à jour")

        # 3. Retour utilisateur selon le cas
        if updated_fields:
            champs = ', '.join(updated_fields)
            send_whatsapp_message(phone_number, f"✅ Facture existante, informations complémentaires ajoutées ou mises à jour : {champs}")
            return
        else:
            send_whatsapp_message(phone_number, "🆕 Nouvelle facture enregistrée avec succès.")

        # 4. Réconciliation de données (exemple)
        result = reconciliation_donnees(facture_data)  # À implémenter selon ta logique

        if result["type"] == "unique":
            message = (
                f"✅ Ticket bien associé à votre paiement de {result['montant']} € "
                f"chez {result['magasin']} du {result['date']} à {result['heure']}."
            )
            send_whatsapp_message(phone_number, message)
        elif result["type"] == "multiple":
            message = "J’ai trouvé plusieurs transactions proches lors de la réconciliation :\n📌 Lequel correspond à votre ticket ?\n"
            for t in result["transactions"]:
                message += f"🔘 {t['montant']} € – {t['magasin']} – {t['heure']}\n"
            message += "🔘 Aucun de ces choix"
            send_whatsapp_message(phone_number, message)
        else:
            send_whatsapp_message(phone_number, "Aucune transaction correspondante trouvée lors de la réconciliation pour ce ticket.")
    except Exception as e:
        logger.error(f"Erreur lors du traitement et de la réponse: {e}")
        send_whatsapp_message(
            phone_number,
            "Désolé, une erreur est survenue lors du traitement de votre ticket."
        )

@app.post("/webhook/")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """Endpoint pour recevoir les webhooks de Twilio."""
    try:
        data = await request.form()
        data_dict = dict(data)
        logger.info(f"Webhook reçu: {data_dict}")

        # Traiter le message en arrière-plan
        background_tasks.add_task(process_incoming_message, data_dict, background_tasks)

        return JSONResponse(content={"status": "success"})
    except Exception as e:
        logger.error(f"Erreur dans le webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Endpoint pour vérifier l'état de l'application."""
    return {"status": "healthy"}

def reconciliation_donnees(facture_data: dict) -> dict:
    """
    Exemple de fonction de réconciliation de données.
    À remplacer par ta logique métier réelle.
    """
    # Exemple : toujours retourner un cas unique pour tester
    return {
        "type": "unique",
        "montant": facture_data.get("montant_ttc", "??"),
        "magasin": facture_data.get("vendeur_nom", "??"),
        "date": facture_data.get("date_vente", "??"),
        "heure": "??"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv('PORT', 8000)),
        log_level="info"
    )
