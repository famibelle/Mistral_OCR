from b2sdk.v2 import InMemoryAccountInfo, B2Api
import os
import dotenv

dotenv.load_dotenv()


def list_bucket_contents():
    """
    Liste le contenu du bucket Backblaze B2.
    """
    try:
        # Initialiser Backblaze B2
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account(
            "production",
            os.getenv("B2_APPLICATION_KEY_ID"),
            os.getenv("B2_APPLICATION_KEY")
        )
        print("Authentification réussie !")

        b2_bucket = b2_api.get_bucket_by_name(os.getenv("B2_BUCKET_NAME"))

        # Lister les fichiers dans le bucket
        print(f"Contenu du bucket '{b2_bucket.name}':")
        for file_version, _ in b2_bucket.ls():
            print(f"- {file_version.file_name} (taille: {file_version.size} octets)")
    except Exception as e:
        print(f"Erreur lors de la récupération du contenu du bucket: {e}")


if __name__ == "__main__":
    list_bucket_contents()