import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="api/api.env")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

'''
Here, remove comments from the model you'd like to utilize for the chatbot.
Similarly, the URL. Huggingface had an outage and had me using the alternate URL for 
their inference API. 
Comment out the ones you won't use, simple.
'''
HF_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
# HF_MODEL_NAME = "tiiuae/falcon-7b-instruct"
# HF_MODEL_NAME = "gpt2"
# HF_MODEL_NAME = "google/flan-t5-large"


# Centralized configuration for LLM providers
LLM_MODEL_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}"
# LLM_MODEL_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_NAME}"

# Summarization Model
SUMMARIZATION_MODEL = "google/flan-t5-large"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Paths for PDF processing and FAISS index storage
PDF_DIRECTORY = "data/pdfs"
FAISS_INDEX_PATH = "data/training_index.faiss"
TRAINING_SENTENCES_PATH = "data/training_sentences.npy"

# Database Path
DATABASE_PATH = "data/database.db"

# HR Support Email
HR_EMAIL = "support@nexa.com"