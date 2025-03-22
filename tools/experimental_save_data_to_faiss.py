import sys
import os
import PyPDF2
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
import requests

# Add the root directory to the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from config import (
    HUGGINGFACE_API_KEY,
    SUMMARIZATION_MODEL,
    EMBEDDING_MODEL,
    PDF_DIRECTORY,
    FAISS_INDEX_PATH,
    TRAINING_SENTENCES_PATH,
)

# Hugging Face API URL and headers
API_URL = f"https://api-inference.huggingface.co/models/{SUMMARIZATION_MODEL}"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Initialize the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Function to summarize text using Hugging Face API
def summarize_text(text, max_length=130, min_length=30):
    """
    Summarizes the given text using the Hugging Face Inference API.
    """
    if len(text.split()) > 1024:  # Truncate text if too long
        text = " ".join(text.split()[:1024])
    payload = {"inputs": text, "parameters": {"max_length": max_length, "min_length": min_length, "do_sample": False}}
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        summary = response.json()

        # Log the full API response for debugging
        print(f"API Response: {summary}")

        # Check if 'summary_text' exists in the response
        if isinstance(summary, list) and "summary_text" in summary[0]:
            return summary[0]["summary_text"]
        elif isinstance(summary, list) and "generated_text" in summary[0]:
            return summary[0]["generated_text"]
        else:
            print(f"Unexpected API response structure: {summary}")
            return text  # Return original text if the response is invalid

    except Exception as e:
        print(f"Error summarizing text: {e}")
        return text  # Return original text if summarization fails

# Function to process PDFs and generate training sentences
def generate_training_sentences(pdf_directory):
    training_sentences = []
    pdf_files = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith(".pdf")]

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        paragraphs = text.split("\n\n")  # Split text into paragraphs

        for paragraph in paragraphs:
            if paragraph.strip():  # Skip empty paragraphs
                summarized_text = summarize_text(paragraph)
                training_sentences.extend(sent_tokenize(summarized_text))  # Split into sentences

    return training_sentences

# Main function to generate FAISS index
def main():
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    print("Generating training sentences from PDFs...")
    training_sentences = generate_training_sentences(PDF_DIRECTORY)
    training_sentences = [sentence.strip() for sentence in training_sentences if sentence.strip()]  # Clean sentences

    print("Generating embeddings...")
    embeddings = embedding_model.encode(training_sentences)

    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index and training sentences
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(TRAINING_SENTENCES_PATH, training_sentences)

    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"Training sentences saved to {TRAINING_SENTENCES_PATH}")

if __name__ == "__main__":
    main()