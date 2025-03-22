import logging
import os
from datetime import datetime

def setup_logger(name="NEXA_HR_Chatbot", log_dir="logs", level=logging.INFO):
    """
    Sets up a logger with the specified name, log directory, and logging level.
    Each session will have a unique log file based on the current timestamp.
    """
    logger = logging.getLogger(name)

    # Check if the logger already has handlers to avoid duplicates
    if logger.hasHandlers():
        return logger

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate a unique log file name using the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger.setLevel(level)

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Stream handler (for console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger