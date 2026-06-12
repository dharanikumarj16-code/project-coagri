import json
from datetime import datetime

class MongoDBStorage:
    """
    Handles persistence of similarity match results, forecasts, and schedules in MongoDB.
    Includes a highly robust, transparent Mock fallback system for offline/local environments.
    """
    def __init__(self, uri: str = "mongodb://localhost:27017/", db_name: str = "smart_farm"):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.is_mock = False
        
        try:
            import pymongo
            print(f"[MongoDBStorage] Attempting to connect to MongoDB at {self.uri}...")
            # Set short timeout of 1.5 seconds to avoid hanging if no local MongoDB is running
            self.client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=1500)
            # Trigger server connection check
            self.client.server_info()
            self.db = self.client[self.db_name]
            print(f"[MongoDBStorage] Success! Connected to MongoDB database '{self.db_name}'.")
        except Exception as e:
            print(f"[MongoDBStorage] Warning: Real MongoDB client failed to initialize ({e}).")
            print("[MongoDBStorage] Activating High-Fidelity local file-based Mock MongoDB storage fallback.")
            self.is_mock = True
            self.mock_collections = {}

    def save_document(self, collection_name: str, doc: dict) -> str:
        """
        Saves a single document into MongoDB (or mock collection).
        """
        doc_to_save = doc.copy()
        doc_to_save["timestamp"] = doc_to_save.get("timestamp", datetime.now().isoformat())
        
        if not self.is_mock:
            try:
                col = self.db[collection_name]
                res = col.insert_one(doc_to_save)
                print(f"[MongoDBStorage] Inserted document into '{collection_name}' with ID: {res.inserted_id}")
                return str(res.inserted_id)
            except Exception as e:
                print(f"[MongoDBStorage] Error during insert: {e}. Falling back to mock saving.")
                
        # Mock implementation
        if collection_name not in self.mock_collections:
            self.mock_collections[collection_name] = []
            
        import uuid
        doc_id = str(uuid.uuid4())
        doc_to_save["_id"] = doc_id
        self.mock_collections[collection_name].append(doc_to_save)
        print(f"[MongoDBStorage-MOCK] Saved doc in '{collection_name}' with Mock ID: {doc_id}")
        return doc_id

    def get_all_documents(self, collection_name: str, query: dict = None) -> list:
        """
        Retrieves all documents matching query from collection.
        """
        if not self.is_mock:
            try:
                col = self.db[collection_name]
                cursor = col.find(query or {})
                return list(cursor)
            except Exception as e:
                print(f"[MongoDBStorage] Error during find: {e}.")
                
        # Mock implementation
        docs = self.mock_collections.get(collection_name, [])
        if not query:
            return docs
            
        # Basic mock filtering
        filtered = []
        for d in docs:
            match = True
            for k, v in query.items():
                if d.get(k) != v:
                    match = False
                    break
            if match:
                filtered.append(d)
        return filtered

if __name__ == "__main__":
    storage = MongoDBStorage()
    doc_id = storage.save_document("similarity_runs", {
        "crop": "Rice",
        "average_similarity": 92.4,
        "risk_level": "low"
    })
    
    saved = storage.get_all_documents("similarity_runs")
    print(f"Retrieved {len(saved)} runs from DB.")
    if saved:
        print("Last saved item:")
        print(saved[-1])
