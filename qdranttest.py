# check_qdrant_data.py (or qdranttest.py)
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables from the .env file
load_dotenv()

# --- Configuration ---
# Use .env variable for collection name, default to "ihrp_collection" if not set
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "ihrp-rag-ultimate") 
# ---------------------

try:
    # Initialize the client using ENV variables (URL and API Key)
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    print("✅ Qdrant client initialized using environment variables.")

    # 1. Check for collection existence by listing all collections
    collections_response = client.get_collections()
    
    # Check if the collection name is in the list of existing collections
    collection_exists = any(
        collection.name == COLLECTION_NAME 
        for collection in collections_response.collections
    )
    
    if not collection_exists:
        print(f"❌ Collection '{COLLECTION_NAME}' does NOT exist in Qdrant.")
        print("Please ensure your indexing script has run successfully.")
    else:
        # 2. Get detailed information about the collection
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        
        # 3. Check the count of vectors/points
        # The points_count is usually available in the result of get_collection
        point_count = collection_info.points_count
        
        if point_count is None:
             print(f"⚠️ Collection '{COLLECTION_NAME}' exists, but point count could not be retrieved.")
        elif point_count > 0:
            print(f"🎉 Success! Collection '{COLLECTION_NAME}' exists with {point_count} vectors (data points).")
            print("Your data is indexed and ready for retrieval.")
        else:
            print(f"⚠️ Collection '{COLLECTION_NAME}' exists but contains 0 vectors.")
            print("You need to run your data indexing/upload script to populate the database.")

except Exception as e:
    print("🛑 An error occurred while connecting or checking Qdrant.")
    print(f"Error: {e}")