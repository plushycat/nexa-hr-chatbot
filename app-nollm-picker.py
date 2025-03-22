import os
import uuid
import sqlite3
from dotenv import load_dotenv
import streamlit as st
from utils.pdf_handler import handle_multiple_pdfs, create_pdf_embeddings, combine_indices
from backend.chatbot import handle_user_message
from utils.logger import setup_logger
import faiss
import numpy as np
from config import LLM_MODEL_URL, DATABASE_PATH, HUGGINGFACE_API_KEY

# Load environment variables
load_dotenv(dotenv_path="api/api.env")

# Validate Hugging Face API key
if not HUGGINGFACE_API_KEY:
    st.error("Hugging Face API key is not set. Please check your .env file.")
    raise ValueError("Hugging Face API key is not set.")
else:
    print(f"API Key Loaded, Enjoy using NEXA!")  # Log only part of the key for security

# Initialize logger with a unique log file for each session
logger = setup_logger()

logger.info("NEXA HR Chatbot application started.")

# Load pre-trained FAISS index and sentences
if "training_index" not in st.session_state:
    try:
        st.session_state["training_index"] = faiss.read_index("data/training_index.faiss")
        st.session_state["training_sentences"] = np.load("data/training_sentences.npy", allow_pickle=True)
        logger.info("Pre-trained FAISS index and sentences loaded successfully.")
    except FileNotFoundError:
        st.session_state["training_index"] = None
        st.session_state["training_sentences"] = None
        logger.error("Pre-trained FAISS index or sentences not found. Ensure the files exist in the 'data' directory.")
    except Exception as e:
        st.session_state["training_index"] = None
        st.session_state["training_sentences"] = None
        logger.error(f"Failed to load pre-trained FAISS index or sentences: {e}")

training_index = st.session_state["training_index"]
training_sentences = st.session_state["training_sentences"]

# Initialize session state variables
if "combined_index" not in st.session_state:
    st.session_state["combined_index"] = training_index  # Default to pre-trained index
if "combined_sentences" not in st.session_state:
    st.session_state["combined_sentences"] = training_sentences  # Default to pre-trained sentences
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())  # Generate a unique session ID

# SQLite database functions
def init_chat_db(db_path="data/chat_history.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            bot_response TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_chat_message(session_id, user_message, bot_response, db_path="data/chat_history.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_history (session_id, user_message, bot_response)
        VALUES (?, ?, ?)
    """, (session_id, user_message, bot_response))
    conn.commit()
    conn.close()

def get_chat_history(session_id, db_path="data/chat_history.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_message, bot_response FROM chat_history
        WHERE session_id = ?
    """, (session_id,))
    history = cursor.fetchall()
    conn.close()
    return history

# Initialize the chat database
init_chat_db()

# Set page configuration
st.set_page_config(page_title="NEXA HR Chatbot", layout="wide")

# Apply custom CSS
with open("styles/theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Create a content wrapper to push content above the footer
content_wrapper = st.container()

with content_wrapper:
    # Sidebar for navigation
    st.sidebar.title("NEXA HR Chatbot")
    page = st.sidebar.selectbox("Navigate", ["Home", "Upload Documents", "Chat with NEXA"])

    # Home Page
    if page == "Home":
        st.title("Welcome to NEXA HR Chatbot")
        st.write("""
            NEXA HR Chatbot is designed to assist you with your HR-related queries and document QnA.
            Navigate through the options in the sidebar to get started.
        """)

    # Upload Documents Page
    elif page == "Upload Documents":
        st.title("Upload Documents")
        st.write("Does NEXA not have information on something, but you have a document,\nand you're lazy to skim through? Upload it here!")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            try:
                # Process uploaded PDFs
                user_text = handle_multiple_pdfs(uploaded_files)
                user_sentences, user_embeddings = create_pdf_embeddings(user_text)

                # Combine with pre-trained FAISS index
                if training_index is not None and training_sentences is not None:
                    combined_index, combined_sentences = combine_indices(
                        training_index, training_sentences, user_embeddings, user_sentences
                    )
                    st.session_state["combined_index"] = combined_index
                    st.session_state["combined_sentences"] = combined_sentences
                    logger.info("Combined FAISS index created successfully.")
                    st.success("PDF files processed and combined with pre-trained data successfully!")
                else:
                    st.session_state["combined_index"] = None
                    st.session_state["combined_sentences"] = None
                    logger.warning("Pre-trained FAISS index not available. Using only user-uploaded data.")
                    st.success("PDF files processed successfully, but no pre-trained data was found.")

            except Exception as e:
                logger.error(f"Error processing uploaded PDFs: {e}")
                st.error("Failed to process the uploaded PDFs. Please ensure they are valid.")

    # Chat with NEXA Page
    elif page == "Chat with NEXA":
        st.title("Chat with NEXA")
        st.subheader("Ask, and NEXA will answer.")

        # Initialize the chatbot's greeting message if it's a new session
        if not get_chat_history(st.session_state["session_id"]):
            save_chat_message(
                st.session_state["session_id"],
                user_message=None,
                bot_response="Hello! I am NEXA, your HR assistant. How can I help you today?"
            )

        # Retrieve chat history for the current session
        chat_history = get_chat_history(st.session_state["session_id"])

        # Create a container for the chat messages
        chat_container = st.container()

        # Display previous chat messages above the input field
        with chat_container:
            st.markdown("---")
            for user_message, bot_response in chat_history:
                if user_message:
                    st.markdown(f"""
                    <div class="chat-message user">
                        <p>{user_message}</p>
                    </div>
                    """, unsafe_allow_html=True)
                if bot_response:
                    st.markdown(f"""
                    <div class="chat-message bot">
                        <p>{bot_response}</p>
                    </div>
                    """, unsafe_allow_html=True)

        # User Input
        st.markdown("---")
        user_input = st.text_area("Type your message here:", key="user_input", height=100)

        # Handle Button Click
        if st.button("Send"):
            if user_input.strip():
                try:
                    print(f"User Input: {user_input}")  # Debug Print

                    # Use combined data if available, otherwise fallback to pre-trained data
                    combined_index = st.session_state.get("combined_index", training_index)
                    combined_sentences = st.session_state.get("combined_sentences", training_sentences)

                    if combined_index is None or combined_sentences is None:
                        raise ValueError("Pre-trained data is not available. Please upload documents.")

                    # Process the user input
                    bot_response = handle_user_message(
                        user_input,
                        combined_index,
                        combined_sentences,
                        LLM_MODEL_URL
                    )
                    print(f"Bot Response: {bot_response}")  # Debug Print

                    # Save the chat message to the database
                    save_chat_message(st.session_state["session_id"], user_input, bot_response)

                    # Render the new message inline at the top of the chat container
                    with chat_container:
                        st.markdown(f"""
                        <div class="chat-message user">
                            <p>{user_input}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="chat-message bot">
                            <p>{bot_response}</p>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    print(f"Error: {e}")  # Debug Print
                    logger.error(f"Error handling user message: {e}")
                    st.error("Failed to process your query. Please try again.")

footer_html = """
    <div class="footer">
        <p>Contact us: <a href="mailto:support@nexa.com">support@nexa.com</a></p>
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)