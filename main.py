import base64
import os
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
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
import asyncio
import imagehash
from PIL import Image
import locale

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

# Configuration Render/PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Initialiser les clients
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

from b2sdk.v2 import InMemoryAccountInfo, B2Api
from io import BytesIO

# Initialisation du client B2 (Backblaze)
B2_APPLICATION_KEY_ID = os.getenv("B2_APPLICATION_KEY_ID")
B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME")

info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY)
b2_bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)


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

class QueryRequest(BaseModel):
    question: str

def upload_image_to_b2(image_bytes: bytes, filename: str) -> str:
    """
    Upload l'image sur Backblaze B2 et retourne l'URL publique.
    """
    file_info = b2_bucket.upload_bytes(
        image_bytes,
        filename,
        content_type="image/jpeg"
    )
    url = f"https://f002.backblazeb2.com/file/{B2_BUCKET_NAME}/{filename}"
    logger.info(f"Image uploadée sur B2: {url}")
    return url


def encode_image(image_bytes: bytes) -> str:
    """Encode des bytes d'image en base64."""
    return base64.b64encode(image_bytes).decode("utf-8")

def process_image(image_bytes: bytes) -> dict:
    """Traite une image de facture (en bytes) et extrait les informations."""
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
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
- numero_facture (string)
- date_emission (ISO YYYY-MM-DD)
- vendeur_nom (string)
- vendeur_adresse (string)
- vendeur_siret (string)
- vendeur_tva (string)
- client_nom (string)
- client_adresse (string)
- description : concatène toutes les lignes d'articles en une seule chaîne de texte lisible (exemple: "Produit A x2 - 10€ HT, Produit B x1 - 5€ HT")
- date_vente (ISO YYYY-MM-DD)
- heure (HH:MM)
- prix_unitaire_ht (float)
- quantite (int)
- taux_tva (string, ex "20%")
- montant_ht (float)
- montant_tva (float)
- montant_ttc (float)
- conditions_paiement (string)
- mentions_legales (string)

**
Assure-toi que tous les montants soient des nombres (float) et que le JSON soit strictement valide.
Le JSON doit être à plat, chaque champ doit être une clé de premier niveau.**
Assure-toi que le JSON est valide et que les champs sont bien formatés.
"""),
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        response_dict = json.loads(chat_response.choices[0].message.content)
        response_dict['image_path'] = None
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
            to=to
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
            # Aucun média : traiter comme une question texte
            if body:
                logger.info(f"Message texte reçu : {body}")
                # Appel interne à la fonction de requête NL→SQL
                try:
                    # On simule un appel à l'endpoint /query/
                    req = QueryRequest(question=body)
                    result = asyncio.run(query_factures(req))  # Appel direct à la fonction FastAPI
                    # Formatage de la réponse pour WhatsApp
                    if "clarification" in result:
                        send_whatsapp_message(data['From'], result["clarification"])
                    elif result["results"]:
                        # Limite à 5 résultats pour WhatsApp
                        lines = []
                        for row in result["results"][:5]:
                            resume = ", ".join(f"{k}: {v}" for k, v in row.items() if k not in ("image_path", "created_at"))
                            lines.append(f"- {resume}")
                        response_text = "Voici les résultats trouvés :\n" + "\n".join(lines)
                        if len(result["results"]) > 5:
                            response_text += f"\n...et {len(result['results'])-5} autres résultat(s)."
                        send_whatsapp_message(data['From'], response_text)
                    else:
                        send_whatsapp_message(data['From'], "Aucune transaction trouvée pour votre question.")
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de la question NL : {e}")
                    send_whatsapp_message(data['From'], "Désolé, je n'ai pas pu traiter votre question.")
            else:
                send_whatsapp_message(data['From'], "Merci d'envoyer une image de ticket ou une question sur vos transactions.")
            return

        # Télécharger l'image (cas image)
        media_url = data[media_url_key]
        response = requests.get(media_url)
        response.raise_for_status()
        image_bytes = response.content

        # Traiter l'image en arrière-plan
        background_tasks.add_task(process_and_respond, data['From'], image_bytes)

        # Enregistrer l'image sur Backblaze B2
        phash = compute_perceptual_hash_from_bytes(image_bytes)
        filename = f"{phash}_facture_{int(time.time())}.jpg"
        image_url = upload_image_to_b2(image_bytes, filename)
    except Exception as e:
        logger.error(f"Erreur lors du traitement du message entrant: {e}")

def process_and_respond(phone_number: str, image_bytes: bytes) -> None:
    try:
        send_whatsapp_message(phone_number, "Merci pour votre envoi ! Je m'en occupe... ⏳")
        facture_data = process_image(image_bytes)
        logger.info(f"Données extraites: {facture_data}")
        updated_fields = check_and_update_database(facture_data)
        logger.info("Base de données mise à jour")

        montant = facture_data.get("montant_ttc", "??")
        numero_facture = facture_data.get("numero_facture", "??")
        date_vente = facture_data.get("date_vente", "??")
        heure = facture_data.get("heure", "??")

        # Formatage de la date et du jour
        try:
            dt = datetime.strptime(date_vente, "%Y-%m-%d")
            jour = dt.strftime("%A").capitalize()
            date_str = dt.strftime("%d/%m/%Y")
        except Exception:
            jour = "?"
            date_str = date_vente

        heure_str = heure if heure and heure != "??" else "?"

        # Si tous les champs sont nouveaux, c'est une nouvelle facture
        if len(updated_fields) == len(facture_data.keys()):
            send_whatsapp_message(
                phone_number,
                f"🆕 Nouvelle facture détectée numéro {numero_facture} le {jour} {date_str} à {heure_str}."
            )
        else:
            send_whatsapp_message(
                phone_number,
                f"✅ Cette facture {numero_facture} du {jour} {date_str} à {heure_str} existe déjà. Je vais l'enrichir avec les nouvelles informations que vous avez données."
            )
            champs = "\n".join(
                f"• {field} : {facture_data.get(field, '')}"
                for field in updated_fields if field not in ("image_path", "image_hash")
            )
            if champs:
                send_whatsapp_message(
                    phone_number,
                    f"Informations ajoutées ou modifiées :\n{champs}"
                )
            else:
                send_whatsapp_message(
                    phone_number,
                    f"Tout est déjà à jour pour la facture {numero_facture}"
                )
        return
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

@app.post("/query/")
async def query_factures(req: QueryRequest):
    """
    Endpoint de requêtes en langage naturel.
    Transforme la question en SQL (via un modèle texte) puis exécute
    strictement ce SQL SELECT sur la base SQLite 'factures.db'.
    Aucun traitement d'image n'est effectué ici.
    """
    # 1. Décrire le schéma à passer au LLM
    schema = """
Table Factures (PostgreSQL) :
facture_id, numero_facture, date_emission, vendeur_nom, vendeur_adresse,
vendeur_siret, vendeur_tva, client_nom, client_adresse, description,
date_vente, prix_unitaire_ht, quantite, taux_tva, montant_ht, montant_tva,
montant_ttc, conditions_paiement, mentions_legales, image_path, created_at
"""
    prompt = (
        f"{schema}\n"
        "Génère uniquement une requête SQL SELECT valide (sans explication), "
        "en sélectionnant, sauf mention contraire, les colonnes montant_ttc, date_vente, heure et description, "
        "pour répondre à :\n"
        f"\"{req.question}\""
    )

    # 2. Génération du SQL avec un modèle texte
    llm_resp = mistral_client.chat.complete(
        model="mistral-large-latest",               # modèle texte
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "text"},
        temperature=0
    )
    sql = llm_resp.choices[0].message.content.strip("```sql").strip("```").strip()
    logger.info(f"SQL générée par LLM : {sql}")

    # 3. Sécurité : n’autoriser que des SELECT
    if not sql.lower().startswith("select"):
        raise HTTPException(400, "Seules les requêtes SELECT sont autorisées.")

    # 4. Exécution sur la base factures.db
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            cols = result.keys()
    except Exception as e:
        logger.error(f"Erreur SQL ({sql}): {e}")
        raise HTTPException(500, "Erreur lors de l'exécution de la requête SQL.")

    # 5. Retour JSON
    results = [dict(zip(cols, row)) for row in rows]
    # Log des résultats
    if not results:
        logger.info("Aucun résultat trouvé pour la requête SQL.")
    else:
        logger.info(f"Résultats de la requête : {results}")    

    # Si trop de colonnes (ex : SELECT *), demander une clarification
    if len(cols) > 6:  # seuil à ajuster selon ton besoin
        return {
            "query": sql,
            "results": [],
            "clarification": (
                "Votre question est trop large et retourne trop d'informations. "
                "Pouvez-vous préciser les champs ou le type de transaction que vous souhaitez obtenir ?"
            )
        }

    return {"query": sql, "results": results}

def check_and_update_database(data: dict) -> list:
    """
    Insère ou met à jour une facture dans la base PostgreSQL selon le quadruplet
    (date_vente, heure, montant_ht, numero_facture) comme identifiant unique.
    Retourne la liste des champs mis à jour ou ajoutés.
    """
    updated_fields = []
    with engine.begin() as conn:
        # Création de la table si besoin (PostgreSQL)
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS Factures (
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
                prix_unitaire_ht REAL,
                quantite INTEGER,
                taux_tva TEXT,
                montant_ht REAL,
                montant_tva REAL,
                montant_ttc REAL,
                conditions_paiement TEXT,
                mentions_legales TEXT,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date_vente, heure, montant_ht, numero_facture)
            )
        '''))

        # Recherche d'une facture existante sur le quadruplet
        result = conn.execute(
            text("""
                SELECT * FROM Factures
                WHERE date_vente = :date_vente AND heure = :heure AND montant_ht = :montant_ht AND numero_facture = :numero_facture
            """),
            {
                "date_vente": data.get('date_vente'),
                "heure": data.get('heure'),
                "montant_ht": data.get('montant_ht'),
                "numero_facture": data.get('numero_facture'),
            }
        )
        existing = result.fetchone()
        cols = result.keys()

        if not existing:
            # Nouvelle facture : INSERT
            conn.execute(
                text('''
                    INSERT INTO Factures (
                        numero_facture, date_emission, vendeur_nom, vendeur_adresse,
                        vendeur_siret, vendeur_tva, client_nom, client_adresse,
                        description, date_vente, heure, prix_unitaire_ht, quantite,
                        taux_tva, montant_ht, montant_tva, montant_ttc,
                        conditions_paiement, mentions_legales,
                        image_path, created_at
                    ) VALUES (
                        :numero_facture, :date_emission, :vendeur_nom, :vendeur_adresse,
                        :vendeur_siret, :vendeur_tva, :client_nom, :client_adresse,
                        :description, :date_vente, :heure, :prix_unitaire_ht, :quantite,
                        :taux_tva, :montant_ht, :montant_tva, :montant_ttc,
                        :conditions_paiement, :mentions_legales,
                        :image_path, :created_at
                    )
                '''), {
                    "numero_facture": data.get('numero_facture'),
                    "date_emission": data.get('date_emission'),
                    "vendeur_nom": data.get('vendeur_nom'),
                    "vendeur_adresse": data.get('vendeur_adresse'),
                    "vendeur_siret": data.get('vendeur_siret'),
                    "vendeur_tva": data.get('vendeur_tva'),
                    "client_nom": data.get('client_nom'),
                    "client_adresse": data.get('client_adresse'),
                    "description": data.get('description'),
                    "date_vente": data.get('date_vente'),
                    "heure": data.get('heure'),
                    "prix_unitaire_ht": data.get('prix_unitaire_ht'),
                    "quantite": data.get('quantite'),
                    "taux_tva": data.get('taux_tva'),
                    "montant_ht": data.get('montant_ht'),
                    "montant_tva": data.get('montant_tva'),
                    "montant_ttc": data.get('montant_ttc'),
                    "conditions_paiement": data.get('conditions_paiement'),
                    "mentions_legales": data.get('mentions_legales'),
                    "image_path": data.get('image_path'),
                    "created_at": data.get('created_at', datetime.now().isoformat())
                }
            )
            updated_fields = list(data.keys())
        else:
            # Facture existante : UPDATE si nouveaux champs
            for idx, col in enumerate(cols):
                if col in ("facture_id", "date_vente", "heure", "montant_ht", "numero_facture"):
                    continue
                new_value = data.get(col)
                old_value = existing[idx]
                if new_value not in (None, "") and str(new_value) != str(old_value):
                    updated_fields.append(col)
            if updated_fields:
                update_fields = ", ".join([f"{col}=:{col}" for col in updated_fields])
                update_sql = f'''
                    UPDATE Factures SET {update_fields}
                    WHERE date_vente = :date_vente AND heure = :heure AND montant_ht = :montant_ht AND numero_facture = :numero_facture
                '''
                params = {col: data.get(col) for col in updated_fields}
                params.update({
                    "date_vente": data.get('date_vente'),
                    "heure": data.get('heure'),
                    "montant_ht": data.get('montant_ht'),
                    "numero_facture": data.get('numero_facture')
                })
                conn.execute(text(update_sql), params)
    return updated_fields

def format_date_heure(date_iso: str, heure: str) -> str:
    try:
        locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
    except:
        pass
    try:
        dt = datetime.strptime(date_iso, "%Y-%m-%d")
        date_str = dt.strftime("%d %B %Y")
        date_str = date_str[0:3] + date_str[3:].lower()
    except Exception:
        date_str = date_iso

    if heure and heure != "??":
        heure_str = heure.replace(":", "h")
        return f"le {date_str} à {heure_str}"
    else:
        return f"le {date_str}"

def compute_perceptual_hash_from_bytes(image_bytes: bytes) -> str:
    try:
        img = Image.open(BytesIO(image_bytes))
        phash = imagehash.phash(img)
        return str(phash)
    except Exception as e:
        logger.error(f"Erreur lors du calcul du perceptual hash: {e}")
        return "nohash"

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv('PORT', 10000)),
        log_level="info"
    )
