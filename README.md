# NEXA HR Chatbot

NEXA HR Chatbot is a Streamlit-based application designed to assist HR teams by leveraging pre-trained language models (LLMs) and FAISS for efficient document-based question answering. This chatbot provides quick and accurate responses to HR-related queries by indexing and searching through HR documents.

---

## Features

- **Document-Based Q&A**: Uses FAISS for fast and scalable document indexing and retrieval.
- **Pre-trained LLMs**: Integrates with Hugging Face models for natural language understanding.
- **Streamlit Interface**: User-friendly web interface for seamless interaction.
- **Customizable**: Easily configurable to adapt to different HR datasets.
- **Logging**: Comprehensive logging for debugging and monitoring.

---

## Project Structure

```
nexa-hr-chatbot/
│
├── src/
│   ├── app.py                  # Main Streamlit application
|   ├── app-nollm-picker.py     # Alternative version without the llm picker (if you want to use only one llm)
|   ├── config.py              # Configuration file with globally applicable variables
|   ├── requirements.txt        # Install with pip install -r requirements.txt
│   ├── tools/
│   │   ├── save_data_to_faiss.py       # Script to index data into FAISS
│   │   ├── verify_api_key.py           # Script to validate Hugging Face API key
│   │   ├── experimental_save_data_to_faiss.py  # Experimental FAISS indexing
│   ├── data/                   # Directory for HR documents and indexed data
│   ├── logs/                   # Directory for log files
│   └── README.md               # Project documentation
```

---

## Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/nexa-hr-chatbot.git
    cd nexa-hr-chatbot/src
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Hugging Face API Key**:
    - Follow this [tutorial](https://huggingface.co/docs/hub/security-tokens) to create your API key.
    - Save the key in an environment variable (.env file) in the api/ directory:
      ```bash
      export HUGGINGFACE_API_KEY=your_api_key
      ```

4. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

---

## Features

1. General QnA Chatbot for HR-related queries
2. Can add documents to expand chatbot's knowledge base (optional)
3. Sample documents are in the `data` directory
4. Chatbot uses information from the sample documents. 
5. Script available to generate your own training data
6. Robust logging system for debugging.

---

## Configuration

- **Hugging Face API Key**: Set the `HUGGINGFACE_API_KEY` environment variable.
- **FAISS Index Path**: Update the `FAISS_INDEX_PATH` in the configuration file if needed.
- **Logging Level**: Modify the logging level in `logging_config.json`.

---

## Key Files

- **`app.py`**: Main application file for the Streamlit interface.
- **`save_data_to_faiss.py`**: Script to index HR documents into FAISS.
- **`verify_api_key.py`**: Script to validate the Hugging Face API key.
- **`experimental_save_data_to_faiss.py`**: Experimental script for advanced FAISS indexing.

---

## Logging

Logs are stored in the `logs/` directory. The logging configuration can be customized in `logging_config.json`.

---

## Future Enhancements

- Add support for multilingual queries.
- Integrate with additional document storage systems.
- Enhance the chatbot's conversational capabilities.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or support, contact [plushycat@protonmail.com].

---

## Scripts in the 'tools' directory:

### `save_data_to_faiss.py`
- **Purpose**: Index HR documents into a FAISS database for efficient retrieval.
- **Usage**:
  ```bash
  python tools/save_data_to_faiss.py --data_dir ./data --index_path ./models/faiss_index
  ```
- **Parameters**:
  - `--data_dir`: Path to the directory containing HR documents.
  - `--index_path`: Path to save the FAISS index.

### `verify_api_key.py`
- **Purpose**: Validate the Hugging Face API key to ensure proper configuration.
- **Usage**:
  ```bash
  python tools/verify_api_key.py
  ```
- **Output**: Prints whether the API key is valid or not.

### `experimental_save_data_to_faiss.py`
- **Purpose**: Experimental script for advanced FAISS indexing with additional features.
- **Usage**:
  ```bash
  python tools/experimental_save_data_to_faiss.py --data_dir ./data --index_path ./models/faiss_index --advanced_mode
  ```
- **Parameters**:
  - `--data_dir`: Path to the directory containing HR documents.
  - `--index_path`: Path to save the FAISS index.
  - `--advanced_mode`: Enables experimental features.
