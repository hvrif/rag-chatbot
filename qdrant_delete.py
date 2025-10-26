import os
from dotenv import load_dotenv # <-- Import this
from qdrant_client import QdrantClient

# Load environment variables from the .env file
load_dotenv() # <-- Add this line!

# The client will now use the correct URL and API key from your .env file
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "ihrp-rag-ultimate"

# Add a print statement to confirm the URL is being loaded correctly
print(f"Attempting to delete collection on URL: {os.getenv('QDRANT_URL')}")

# Delete the collection
try:
    client.delete_collection(collection_name=collection_name)
    print(f"🎉 Success! Collection '{collection_name}' deleted.")
except Exception as e:
    print(f"🛑 Error during collection deletion: {e}")