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

class QueryRequest(BaseModel):
    question: str

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
    On considère désormais comme identifiant unique d'une facture
    le quadruplet (date_vente, heure, montant_ht, montant_ttc).
    """
    updated_fields = []
    conn = sqlite3.connect('factures.db')
    cursor = conn.cursor()

    # 1) Créer la table si elle n'existe pas, avec UNIQUE sur le quadruplet
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Factures (
            facture_id INTEGER PRIMARY KEY AUTOINCREMENT,
            numero_facture TEXT,
            date_emission TEXT,
            vendeur_nom TEXT,
            vendeur_adresse TEXT,
            vendeur_siret TEXT,
            vendeur_tva TEXT,
            client_nom TEXT,
            client_adresse TEXT,
            description TEXT,
            date_vente TEXT,
            heure TEXT,
            prix_unitaire_ht REAL,
            quantite INTEGER,
            taux_tva TEXT,
            montant_ht REAL,
            montant_tva REAL,
            montant_ttc REAL,
            conditions_paiement TEXT,
            mentions_legales TEXT,
            image_path TEXT,
            image_hash TEXT,
            created_at TEXT,
            UNIQUE(date_vente, heure, montant_ht, montant_ttc)
        )
    ''')

    # 2) Vérifier existence sur (date_vente, heure, montant_ht, montant_ttc)
    cursor.execute(
        """
        SELECT * FROM Factures
        WHERE date_vente = ?
          AND heure = ?
          AND montant_ht = ?
          AND montant_ttc = ?
        """,
        (
            data.get('date_vente'),
            data.get('heure'),
            data.get('montant_ht'),
            data.get('montant_ttc'),
        )
    )
    existing = cursor.fetchone()
    cols = [d[0] for d in cursor.description]

    new_path = data.get('image_path')
    new_hash = compute_perceptual_hash(new_path) if new_path else None

    if not existing:
        # Nouvelle facture : INSERT
        paths = [new_path] if new_path else []
        hashes = [new_hash] if new_hash else []
        cursor.execute('''
            INSERT INTO Factures (
                numero_facture, date_emission, vendeur_nom, vendeur_adresse,
                vendeur_siret, vendeur_tva, client_nom, client_adresse,
                description, date_vente, heure, prix_unitaire_ht, quantite,
                taux_tva, montant_ht, montant_tva, montant_ttc,
                conditions_paiement, mentions_legales,
                image_path, image_hash, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            data.get('date_vente'),       # 10ᵉ valeur
            data.get('heure'),            # 11ᵉ valeur – NE PAS OUBLIER
            data.get('prix_unitaire_ht'), # 12ᵉ
            data.get('quantite'),
            data.get('taux_tva'),
            data.get('montant_ht'),
            data.get('montant_tva'),
            data.get('montant_ttc'),
            data.get('conditions_paiement'),
            data.get('mentions_legales'),
            json.dumps(paths),
            json.dumps(hashes),
            data.get('created_at', datetime.now().isoformat())
        ))
        logger.info("Nouvelle facture insérée.")
    else:
        # Facture existante : UPDATE
        old_paths = json.loads(existing[cols.index('image_path')] or '[]')
        old_hashes = json.loads(existing[cols.index('image_hash')] or '[]')
        if new_hash and new_hash not in old_hashes:
            old_paths.append(new_path)
            old_hashes.append(new_hash)
            updated_fields += ['image_path', 'image_hash']
        # Vérification des autres champs
        for idx, col in enumerate(cols):
            if col in ("facture_id", "date_vente", "heure", "montant_ht", "montant_ttc", "image_path", "image_hash"):
                continue
            new_value = data.get(col)
            old_value = existing[idx]
            if new_value not in (None, "") and str(new_value) != str(old_value):
                updated_fields.append(col)
        # Appliquer la mise à jour
        cursor.execute('''
            UPDATE Factures SET
                numero_facture=?, date_emission=?, vendeur_nom=?, vendeur_adresse=?,
                vendeur_siret=?, vendeur_tva=?, client_nom=?, client_adresse=?,
                description=?, prix_unitaire_ht=?, quantite=?, taux_tva=?,
                montant_ht=?, montant_tva=?, montant_ttc=?, conditions_paiement=?,
                mentions_legales=?, image_path=?, image_hash=?, created_at=?
            WHERE date_vente = ? AND heure = ? AND montant_ht = ? AND montant_ttc = ?
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
            data.get('prix_unitaire_ht'),
            data.get('quantite'),
            data.get('taux_tva'),
            data.get('montant_ht'),         # <-- AJOUTÉ (était manquant)
            data.get('montant_tva'),
            data.get('montant_ttc'),
            data.get('conditions_paiement'),
            data.get('mentions_legales'),
            json.dumps(old_paths),
            json.dumps(old_hashes),
            data.get('created_at', datetime.now().isoformat()),
            # Critère unique
            data.get('date_vente'),
            data.get('heure'),
            data.get('montant_ht'),
            data.get('montant_ttc'),
        ))
        logger.info(f"Facture mise à jour, champs modifiés : {updated_fields}")
    conn.commit()
    conn.close()
    return updated_fields

# --- Dans process_image, pour que image_path soit toujours une liste JSON dans la base ---
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
        # Toujours retourner le chemin de la nouvelle image (pour ajout dans la liste)
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

        montant = facture_data.get("montant_ttc", "??")
        magasin = facture_data.get("vendeur_nom", "??")
        date_vente = facture_data.get("date_vente", "??")
        heure = facture_data.get("heure", "??")
        champs = "\n".join(
            f"• {field} : {facture_data.get(field, '')}"
            for field in updated_fields if field not in ("image_path", "image_hash")
        )

        # Utilise la fonction de formatage pour la date et l'heure
        date_heure_str = format_date_heure(date_vente, heure)

        send_whatsapp_message(
            phone_number,
            f"✅Transaction trouvée! Je l'ai associée à votre paiement de {montant}€ chez {magasin} {date_heure_str}."
        )
        send_whatsapp_message(
            phone_number,
            "Je l'ai associé à la transaction automatiquement."
        )
        if champs:
            send_whatsapp_message(
                phone_number,
                f"J'en ai profité pour rajouter des informations complémentaires à la transaction existante:\n{champs}"
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
Table Factures (SQLite – factures.db) :
facture_id, numero_facture, date_emission, vendeur_nom, vendeur_adresse,
vendeur_siret, vendeur_tva, client_nom, client_adresse, description,
date_vente, prix_unitaire_ht, quantite, taux_tva, montant_ht, montant_tva,
montant_ttc, conditions_paiement, mentions_legales, image_path, created_at
"""
    prompt = (
        f"{schema}\n"
        "Génère uniquement une requête SQL SELECT valide (sans explication), "
        "en sélectionnant par défaut les colonnes montant_ttc, date_vente, heure et description, "
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
        conn = sqlite3.connect('factures.db')
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [col[0] for col in cursor.description]
    except Exception as e:
        logger.error(f"Erreur SQL ({sql}): {e}")
        raise HTTPException(500, "Erreur lors de l'exécution de la requête SQL.")
    finally:
        conn.close()

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

def compute_perceptual_hash(image_path: str) -> str:
    """
    Calcule et retourne le hash perceptuel (pHash) de l'image.
    Utile pour détecter les duplications ou similitudes d'images.
    """
    try:
        img = Image.open(image_path)
        phash = imagehash.phash(img)
        return str(phash)
    except Exception as e:
        logger.error(f"Erreur lors du calcul du perceptual hash pour {image_path}: {e}")
        return ""

def format_date_heure(date_iso: str, heure: str) -> str:
    """
    Transforme '2025-06-14' et '16:14' en 'le 14 juin 2025 à 16h14'
    """
    try:
        # Pour avoir les mois en français
        locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
    except:
        pass  # Sur Windows, il peut être nécessaire d'installer le pack de langue

    try:
        dt = datetime.strptime(date_iso, "%Y-%m-%d")
        date_str = dt.strftime("%d %B %Y")
        # Mettre la première lettre du mois en minuscule
        date_str = date_str[0:3] + date_str[3:].lower()
    except Exception:
        date_str = date_iso

    if heure and heure != "??":
        heure_str = heure.replace(":", "h")
        return f"le {date_str} à {heure_str}"
    else:
        return f"le {date_str}"

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv('PORT', 8000)),
        log_level="info"
    )
