from utils.logger import setup_logger
from utils.llm_handler import query_llm
from utils.pdf_handler import search_pdf_context

logger = setup_logger()

def handle_user_message(user_input, combined_index, combined_sentences, model_name):
    """
    Handles user messages, retrieves context, and generates friendly responses using the Hugging Face Inference API.
    """
    logger.info(f"Received user input: {user_input}")

    # Step 1: Retrieve context using FAISS
    if combined_index is not None and combined_sentences is not None:
        context = search_pdf_context(user_input, combined_index, combined_sentences)
        logger.info(f"Retrieved context: {context}")
    else:
        context = "No relevant context found in the uploaded documents."
        logger.warning("No FAISS index or sentences available.")

    # Step 2: Generate a friendly response using the Hugging Face Inference API
    try:
        if context == "No relevant context found in the uploaded documents.":
            response = (
                "I'm sorry, I couldn't find any relevant information in the uploaded documents. "
                "Could you try rephrasing your question or uploading more documents?"
            )
        else:
            # Use the Hugging Face Inference API to generate a response
            response = query_llm(model_name, user_input, context)
    except Exception as e:
        logger.error(f"Error generating response with LLM: {e}")
        response = (
            "I'm sorry, something went wrong while processing your request. "
            "Please try again later or contact support."
        )

    logger.info(f"Generated response: {response}")
    return response