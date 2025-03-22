import sys
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2

# Add the root directory to the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# Import configuration from the root directory
from config import (
    EMBEDDING_MODEL,
    PDF_DIRECTORY,
    FAISS_INDEX_PATH,
    TRAINING_SENTENCES_PATH,
)

# Load the embedding model from config
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a single PDF file.
    """
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Function to process multiple PDFs
def process_multiple_pdfs(pdf_paths):
    """
    Processes multiple PDF files and returns a list of sentences.
    """
    all_text = ""
    for pdf_path in pdf_paths:
        all_text += extract_text_from_pdf(pdf_path) + "\n"
    return all_text.split("\n")  # Split text into sentences

# Training data (e.g., HR policies, FAQs)
training_sentences = [
    "The leave policy allows for 15 annual leaves, with 5 days carried over each year.",
    "Employees should contact HR at support@nexa.com for any policy-related queries.",
    "Generative AI tools are allowed for learning and enhancement but must be used under strict controls.",
    "NEXA's HR policies ensure equal opportunities and merit-based promotions across the organization.",
    "Leave requests must be submitted via the HR portal with at least 10 business days notice for planned absences.",
    "Maternity leave is fully paid for 26 weeks, with an option for remote work up to 3 hours daily.",
    "Paternity leave provides 60 days of full pay, with an option for remote contributions if needed.",
    "Contract employees receive payments based on milestones defined in their contracts.",
    "The payroll guide includes statutory deductions such as TDS, EPF, ESI, and Professional Tax.",
    "Overtime is compensated at 1.5 times the regular hourly rate for hours worked beyond 45 per week.",
    "NEXA's share-based compensation program includes stock options and performance shares to align employee performance with company growth.",
    "Employees must disclose any potential conflicts of interest promptly to maintain workplace integrity.",
    "Payroll details, including leave balances and statutory deductions, are accessible in real time through the HR portal.",
    "HR maintains a zero-tolerance policy toward harassment and discrimination in the workplace.",
    "Remote work requires using company-authorized VPN services such as ProtonVPN and WindScribe.",
    "Employees must refrain from sharing sensitive company data on external platforms.",
    "Performance bonuses and quarterly incentives are awarded based on individual merit and overall company performance.",
    "In emergency leave situations, employees must notify their supervisor immediately and log the request online.",
]

# Extract sentences from PDFs
pdf_files = [os.path.join(PDF_DIRECTORY, file) for file in os.listdir(PDF_DIRECTORY) if file.endswith(".pdf")]
pdf_sentences = process_multiple_pdfs(pdf_files)

# Combine training data with PDF sentences
all_sentences = training_sentences + pdf_sentences

# Generate embeddings
all_embeddings = embedding_model.encode(all_sentences)

# Create a FAISS index
index = faiss.IndexFlatL2(all_embeddings.shape[1])
index.add(all_embeddings)

# Save the index and sentences
faiss.write_index(index, FAISS_INDEX_PATH)
np.save(TRAINING_SENTENCES_PATH, all_sentences)

print(f"FAISS index saved to {FAISS_INDEX_PATH}")
print(f"Training sentences saved to {TRAINING_SENTENCES_PATH}")