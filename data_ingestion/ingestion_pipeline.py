import os
import pandas as pd 
from dotenv import load_dotenv
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from config.config_loader import config_loader
from utils.model_loader import ModelLoader


class Data_Ingestion:

    def __init__(self):
        "Data transformation and ingestion to vector DB."
        print("Initializing the pipeline of Data ingestion")
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()
        self.config = config_loader()

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

    def _get_csv_path(self):
        "Gets the path of csv file which is iniside the data folder."

        current_dir = os.getcwd()
        # print("CURRENT DIRECTORY: ", current_dir)
        csv_path = os.path.join(current_dir, 'data', 'flipkart_product_review.csv')
        # print("CSV_PATH:", csv_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        
        return csv_path
    
    def _load_csv(self):
        "Loading the csv file."

        df = pd.read_csv(self.csv_path)
        # print("DF: ", df)
        expected_columns = {'product_title', 'rating', 'summary', 'review'}

        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {expected_columns}")
        
        return df
    
    def transform_data(self):
        "Transform data into list of langchain documents."

        product_list =[]

        for _, row in self.product_data.iterrows():
            product_entry ={
            'product_title' : row['product_title'],
            'product_rating' : row['rating'],
            "product_summary" : row['summary'],
            'product_review' : row['review']
            }
            product_list.append(product_entry)

        documents = []
        for row in product_list:
            metadata = {
            'product_title' : row['product_title'],
            'product_rating' : row['product_rating'],
            "product_summary" : row['product_summary']
            }
            doc = Document(page_content=row['product_review'], metadata=metadata)
            documents.append(doc)

        print(f"Transformed {len(documents)} documents.")
        return documents

    def store_in_vector_db(self, documents: List[Document]):
        "Store the documents in AstraDB vector store."
        collection_name = self.config['astra_db']['collection_name']
        vector_store = AstraDBVectorStore(
            embedding=self.model_loader.load_embedding(),
            collection_name=collection_name,
            api_endpoint = self.db_api_endpoint,
            token = self.db_application_token,
            namespace = self.db_keyspace
        )
        inserted_ids = vector_store.add_documents(documents)
        print(f"Succesfully inserted {len(inserted_ids)}")
        return vector_store, inserted_ids
    
    def run_pipeline(self):
        "Data Ingestion pipeline: transforms data and store into vector DB"
        documents = self.transform_data()
        vector_store, inserted_ids = self.store_in_vector_db(documents)

        query = "tell me some low budget headphones"
        result = vector_store.similarity_search(query)

        print(f"\nSample search results for the query: {query}")
        for res in result:
            print(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")


if __name__ == "__main__":
    ingestion = Data_Ingestion()
    ingestion.run_pipeline()