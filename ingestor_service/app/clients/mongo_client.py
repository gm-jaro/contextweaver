from pymongo import MongoClient
from app.config.costants import MONGO_DB_NAME
from app.config.settings import MONGO_URL


class MongoDBClient:
    def __init__(self):
        self.client = MongoClient(MONGO_URL)
        self.db = self.client[MONGO_DB_NAME]

    def get_collection(self, name: str):
        return self.db[name]
