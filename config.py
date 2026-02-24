"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model 


# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""


    # Input documents folder
    DATA_FOLDER = "./data"

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model Configuration (Groq-supported models)
    # Good options: "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"
    # LLM_MODEL = "llama-3.1-8b-instant"
    GROQ_LLM_MODEL = "openai/gpt-oss-120b"
    LLM_MODEL = "openai:gpt-4o"

    # Document Processing
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150

    CLIP_MODEL = "openai/clip-vit-base-patch32"
    TEXT_EMBEDDING = "BAAI/bge-large-en-v1.5"


    @classmethod
    def get_llm_groq(cls):
        """Initialize and return the Groq LLM model"""
        return ChatGroq(
            groq_api_key=cls.GROQ_API_KEY,
            model_name=cls.LLM_MODEL,
            temperature=0
        )
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)

