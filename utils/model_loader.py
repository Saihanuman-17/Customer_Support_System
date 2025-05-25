import os 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config_loader import config_loader

class ModelLoader:
    "Class to load the embedding and llm models"
    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config=config_loader()

    def _validate_env(self):
        "Validates the environment variable"
        required_vars = ['GOOGLE_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing Environment Variable: {missing_vars}")

    def load_embedding(self):
        "Load and returns the embedding model"
        embed_model_name=self.config["embedding_model"]["model"]
        embedding_model = GoogleGenerativeAIEmbeddings(model=embed_model_name)
        return embedding_model
    
    def load_llm(self):
        "Load and returns the LLm model"
        print("Loading LLM model")
        llm_model_name = self.config["llm"]["model_name"]
        gemini_model = ChatGoogleGenerativeAI(model = llm_model_name)
        return gemini_model
