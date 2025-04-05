!pip install llama-index llama-index-llms-gemini llama-index-embeddings-gemini google-generativeai -v
!pip install pypdf -v
from google.colab import userdata
import os

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

import shutil
import os

PERSIST_DIR = "./storage"
if os.path.exists(PERSIST_DIR):
    print(f"Deleting old index directory: {PERSIST_DIR}")
    shutil.rmtree(PERSIST_DIR)
else:
    print("No existing index directory to delete.")

import os
from llama_index.core import SimpleDirectoryReader

os.makedirs("data", exist_ok=True)

DATA_DIR = "./data"

if not os.path.exists(DATA_DIR):
    print(f"Error: Directory '{DATA_DIR}' not found.")
    print("Please create the 'data' directory and upload your files.")
elif not os.listdir(DATA_DIR):
    print(f"Warning: Directory '{DATA_DIR}' is empty.")
    print("Please upload your data files into the 'data' directory.")
else:
    print(f"Loading documents from '{DATA_DIR}'...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

documents = []
if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
    print(f"Loading documents from '{DATA_DIR}'...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

if documents:
    print(f"Successfully loaded {len(documents)} document(s).")
    print("\n--- Sample Text from First Document ---")
    if documents[0].text and len(documents[0].text.strip()) > 0:
         print(documents[0].text[:500].strip() + "...")
         print("-> Text seems to be extracted successfully.")
    else:
         print("-> WARNING: First document appears to have NO extracted text.")
         print("-> Ensure 'pypdf' is installed and the PDF is not image-based or corrupted.")
    print("--- End Sample Text ---")
else:
    print("Loaded 0 documents.")
    if documents:
        print(f"Successfully loaded {len(documents)} document(s).")
    else:
        print("Loaded 0 documents. Check if the files are in the correct format or if the directory is empty.")

import os
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from google.colab import userdata

try:
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in Colab Secrets.")
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    print("Google API Key loaded.")
except Exception as e:
    print(f"Error loading Google API Key from Colab Secrets: {e}")
    print("Please ensure the secret 'GOOGLE_API_KEY' is created and notebook access is enabled.")
    raise SystemExit("API Key configuration failed.")

LLM_MODEL_NAME = "models/gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

print(f"Configuring LLM: {LLM_MODEL_NAME}")
Settings.llm = Gemini(model_name=LLM_MODEL_NAME, api_key=GOOGLE_API_KEY)

print(f"Configuring Embedding Model: {EMBEDDING_MODEL_NAME}")
Settings.embed_model = GeminiEmbedding(model_name=EMBEDDING_MODEL_NAME)

PERSIST_DIR = "./storage"

if 'documents' not in locals() or not documents:
    print("Error: 'documents' variable not found or is empty.")
    print("Please run the Data Loading cell successfully first.")
    raise SystemExit("Document loading failed or skipped.")
else:
    if not os.path.exists(PERSIST_DIR):
        print(f"Creating new index from {len(documents)} documents...")
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        print(f"Index created successfully.")
        print(f"Persisting index to '{PERSIST_DIR}'...")
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("Index persisted.")
    else:
        print(f"Loading existing index from '{PERSIST_DIR}'...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("Index loaded successfully.")


if 'index' not in locals():
    print("Error: 'index' variable not found.")
    print("Please run the Indexing cell successfully first.")
    raise SystemExit("Index creation/loading failed or skipped.")
else:
    print("Creating query engine...")
    query_engine = index.as_query_engine()
    print("Query engine created.")

    print("\n--- Running Sample Query ---")
    sample_query = "What is the main topic discussed in the documents? Be detailed"
    print(f"Query: {sample_query}")

    response = query_engine.query(sample_query)

    print("\nResponse:")
    print(response)
    print("--- End Sample Query ---")


if 'index' not in locals():
    print("Error: 'index' variable not found.")
    print("Please run the Indexing cell successfully first.")
    raise SystemExit("Index creation/loading failed or skipped.")
else:
    print("Creating chat engine...")
    chat_engine = index.as_chat_engine(
        chat_mode='context',
        verbose=True
    )
    print("Chat engine created. Type 'quit' or 'exit' to end the chat.")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Bot: Goodbye!")
                break
            response = chat_engine.chat(user_input)
            print(f"Bot: {response}")
        except EOFError:
            print("\nBot: Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break
