import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import unicodedata
from utils.logger import setup_logger

logger = setup_logger()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a single PDF file and normalize it to handle special characters.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            raw_text = page.extract_text()
            # Normalize text to replace special characters (e.g., ligatures)
            normalized_text = unicodedata.normalize("NFKD", raw_text)
            text += normalized_text
        return text
    except Exception as e:
        raise RuntimeError(f"Error extracting text from PDF: {e}")

def handle_multiple_pdfs(uploaded_files):
    """
    Process multiple PDF files and return the combined text.
    """
    all_text = ""
    for uploaded_file in uploaded_files:
        try:
            text = extract_text_from_pdf(uploaded_file)
            if text.strip():
                all_text += text + "\n"
        except Exception as e:
            raise RuntimeError(f"Error processing file {uploaded_file.name}: {e}")
    return all_text

def create_pdf_embeddings(pdf_text):
    """
    Creates embeddings for the given PDF text.
    """
    sentences = pdf_text.split("\n")
    embeddings = embedding_model.encode(sentences)
    return sentences, embeddings

def search_pdf_context(query, combined_index, combined_sentences, top_k=3):
    """
    Searches the combined FAISS index for the most relevant sentences.
    """
    try:
        query_embedding = embedding_model.encode([query])
        _, indices = combined_index.search(query_embedding, top_k)
        best_matches = [combined_sentences[idx] for idx in indices[0] if idx < len(combined_sentences)]
        logger.info(f"Query: {query}")
        logger.info(f"Retrieved context: {best_matches}")
        return " ".join(best_matches) if best_matches else "No relevant context found."
    except Exception as e:
        logger.error(f"Error searching FAISS index: {e}")
        return "No relevant context found."

def combine_indices(pretrained_index, pretrained_sentences, user_embeddings, user_sentences):
    """
    Combines pre-trained FAISS index with user-uploaded embeddings.
    """
    combined_index = faiss.IndexFlatL2(pretrained_index.d)
    combined_index.add(pretrained_index.reconstruct_n(0, pretrained_index.ntotal))
    combined_index.add(user_embeddings)
    combined_sentences = np.concatenate([pretrained_sentences, user_sentences])
    return combined_index, combined_sentences