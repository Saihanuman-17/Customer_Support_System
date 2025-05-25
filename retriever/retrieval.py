from langchain_astradb import AstraDBVectorStore
from typing import List
from langchain_core.documents import Document
from config.config_loader import config_loader
from utils.model_loader import ModelLoader
from dotenv import load_dotenv
import os 
 

class Retriever:

    def __init__(self):
        self.model_loader = ModelLoader()
        self.config = config_loader()
        self._load_env_variables()
        self.vector_store = None
        self.retriever = None

    def _load_env_variables(self):
        "Load and validate the environment variables"

        load_dotenv()
        required_vars = ['GOOGLE_API_KEY','ASTRA_DB_API_ENDPOINT','ASTRA_DB_APPLICATION_TOKEN','ASTRA_DB_KEYSPACE']
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing Environment variables {missing_vars}")
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.db_api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
        # print("self.db_api_endpoint:", self.db_api_endpoint)
        self.db_application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
        self.db_keyspace = os.getenv('ASTRA_DB_KEYSPACE')


    def load_retriever(self):
        if not self.vector_store:
            collection_name = self.config['astra_db']['collection_name']

            self.vector_store = AstraDBVectorStore(
                    embedding=self.model_loader.load_embedding(),
                    collection_name=collection_name,
                    api_endpoint = self.db_api_endpoint,
                    token = self.db_application_token,
                    namespace = self.db_keyspace
                )
        
        if not self.retriever:
            top_k = self.config['retriever']['top_k'] if "retriever" in self.config else 3
            retriever = self.vector_store.as_retriever(search_kwargs={"k":top_k})
            print("Retriever Loaded Successfully")
            return retriever

    def call_retriever(self, query:str)-> List[Document]:
        retriever = self.load_retriever()
        output = retriever.invoke(query)
        return output



if __name__ == '__main__':
    retriver_obj = Retriever()
    user_query = 'suggest some laptops with good configuration'
    results = retriver_obj.call_retriever(user_query)

    for idx, doc in enumerate(results, 1):
        print(f"Results {idx}: {doc.page_content}\n metadata: {doc.metadata}\n")

    
