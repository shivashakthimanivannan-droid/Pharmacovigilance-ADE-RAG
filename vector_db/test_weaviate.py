import os
import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth

load_dotenv()

url = os.getenv("WEAVIATE_URL")
api_key = os.getenv("WEAVIATE_API_KEY")

print("URL:", url)
print("API key loaded:", bool(api_key))

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=url,
    auth_credentials=Auth.api_key(api_key)
)

print("Connected!")
print("Ready:", client.is_ready())

client.close()