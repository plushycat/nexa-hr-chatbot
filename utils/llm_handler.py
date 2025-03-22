from config import HUGGINGFACE_API_KEY
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests

from utils import logger

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def query_llm(model_name, question, context):
    """
    Queries the Hugging Face Inference API with a question and context for generating a conversational response.
    """
    prompt = (
        f"You are a friendly and helpful HR assistant. Use the following context to answer the question:\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return query_llm_inference_api(model_name, prompt)

def query_llm_inference_api(model_name, prompt):
    """
    Queries the Hugging Face Inference API with a prompt.
    """
    # Use the model_name directly if it's a full URL
    api_url = model_name if model_name.startswith("http") else f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes

        # Extract only the generated text
        generated_text = response.json()[0].get("generated_text", "").strip()

        # Remove the prompt from the response if it is included
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()

        return generated_text
    except requests.exceptions.HTTPError as http_err:
        raise RuntimeError(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        raise RuntimeError(f"Request error occurred: {req_err}")
    except KeyError:
        raise RuntimeError("Unexpected response format from Hugging Face API.")

def classify_intent(user_input, intent_model):
    """
    Classifies the intent of the user input using a text-classification model.
    """
    try:
        result = intent_model(user_input)
        intent = result[0]["label"].lower()  # Ensure the intent matches predefined intents
        return intent
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        return "general_query" 

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a single PDF file.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise RuntimeError(f"Error extracting text from PDF: {e}")

def create_pdf_embeddings(pdf_text):
    """
    Creates embeddings for the given PDF text.
    """
    try:
        sentences = pdf_text.split("\n")
        embeddings = embedding_model.encode(sentences)
        return sentences, embeddings
    except Exception as e:
        raise RuntimeError(f"Error creating PDF embeddings: {e}")

def search_pdf_context(query, combined_index, combined_sentences, top_k=3):
    """
    Searches the combined FAISS index for the most relevant sentences.
    """
    try:
        query_embedding = embedding_model.encode([query])
        _, indices = combined_index.search(query_embedding, top_k)
        best_matches = [combined_sentences[idx] for idx in indices[0]]
        return " ".join(best_matches)  # Combine the top-k sentences into a single context
    except Exception as e:
        return f"Error searching FAISS index: {e}"