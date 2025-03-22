import sys
import os
import requests

# Add the root directory to the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from config import HUGGINGFACE_API_KEY, LLM_MODEL_URL

def verify_api_key():
    """
    Verifies the Hugging Face API key by making a test request to the specified model URL.
    """
    # Use centralized LLM_MODEL_URL and HUGGINGFACE_API_KEY
    api_url = LLM_MODEL_URL
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": "Test prompt"}

    try:
        # Make a POST request to the Hugging Face API
        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            print("API key is valid, and the model is accessible.")
            try:
                print("Response:", response.json())
            except requests.exceptions.JSONDecodeError:
                print("Response is not in JSON format.")
        elif response.status_code == 401:
            print("Unauthorized: The API key is invalid or does not have access to the model.")
        elif response.status_code == 503:
            print("Service Unavailable: The server is temporarily unavailable. Please try again later.")
        else:
            print(f"Error: Received status code {response.status_code}")
            try:
                print("Response:", response.json())
            except requests.exceptions.JSONDecodeError:
                print("Response is not in JSON format.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")

if __name__ == "__main__":
    verify_api_key()