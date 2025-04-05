import os
import logging
import shutil
import sqlite3
import datetime
from google.colab import userdata
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

# Configuration
BASE_URL = "https://testnet.binancefuture.com"
SYMBOL = "BTCUSDT"
FETCH_INTERVAL = 60  # 60 seconds = 1 minute

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API Key from environment
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY not found in Colab Secrets.")
    raise SystemExit("API Key configuration failed.")
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# Setup SQLite database
def setup_database():
    db_connection = sqlite3.connect('prices.db')
    db_cursor = db_connection.cursor()
    db_cursor.execute('''
    CREATE TABLE IF NOT EXISTS PriceHistory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        price REAL NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    db_cursor.execute('''
    CREATE TABLE IF NOT EXISTS OrderBook (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        bids TEXT NOT NULL,
        asks TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    db_cursor.execute('''
    CREATE TABLE IF NOT EXISTS RecentTrades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        trade_id INTEGER NOT NULL,
        price REAL NOT NULL,
        qty REAL NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    db_cursor.execute('''
    CREATE TABLE IF NOT EXISTS PriceChangeStats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        price_change REAL NOT NULL,
        price_change_percent REAL NOT NULL,
        weighted_avg_price REAL NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    db_cursor.execute('''
    CREATE TABLE IF NOT EXISTS CandlestickData (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        open_time INTEGER NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        close_time INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    db_connection.commit()
    return db_connection, db_cursor

db_connection, db_cursor = setup_database()

# Setup directories
def setup_directories():
    PERSIST_DIR = "./storage"
    if os.path.exists(PERSIST_DIR):
        logging.info(f"Deleting old index directory: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)
    else:
        logging.info("No existing index directory to delete.")
    os.makedirs("data", exist_ok=True)
    return PERSIST_DIR

PERSIST_DIR = setup_directories()

# Load documents
def load_documents(data_dir):
    if not os.path.exists(data_dir):
        logging.error(f"Error: Directory '{data_dir}' not found.")
        raise SystemExit("Data directory not found.")
    elif not os.listdir(data_dir):
        logging.warning(f"Warning: Directory '{data_dir}' is empty.")
        raise SystemExit("Data directory is empty.")
    else:
        logging.info(f"Loading documents from '{data_dir}'...")
        documents = SimpleDirectoryReader(data_dir).load_data()
        if documents:
            logging.info(f"Successfully loaded {len(documents)} document(s).")
            logging.info("\n--- Sample Text from First Document ---")
            if documents[0].text and len(documents[0].text.strip()) > 0:
                logging.info(documents[0].text[:500].strip() + "...")
                logging.info("-> Text seems to be extracted successfully.")
            else:
                logging.warning("-> WARNING: First document appears to have NO extracted text.")
                logging.warning("-> Ensure 'pypdf' is installed and the PDF is not image-based or corrupted.")
            logging.info("--- End Sample Text ---")
        else:
            logging.error("Loaded 0 documents. Check if the files are in the correct format or if the directory is empty.")
        return documents

DATA_DIR = "./data"
documents = load_documents(DATA_DIR)

# Configure LLM and Embedding Model
def configure_llm_and_embedding():
    LLM_MODEL_NAME = "models/gemini-2.0-flash"
    EMBEDDING_MODEL_NAME = "models/text-embedding-004"
    logging.info(f"Configuring LLM: {LLM_MODEL_NAME}")
    Settings.llm = Gemini(model_name=LLM_MODEL_NAME, api_key=GOOGLE_API_KEY)
    logging.info(f"Configuring Embedding Model: {EMBEDDING_MODEL_NAME}")
    Settings.embed_model = GeminiEmbedding(model_name=EMBEDDING_MODEL_NAME)

configure_llm_and_embedding()

# Create or load index
def create_or_load_index(documents, persist_dir):
    if not documents:
        logging.error("Error: 'documents' variable not found or is empty.")
        raise SystemExit("Document loading failed or skipped.")
    else:
        if not os.path.exists(persist_dir):
            logging.info(f"Creating new index from {len(documents)} documents...")
            index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )
            logging.info(f"Index created successfully.")
            logging.info(f"Persisting index to '{persist_dir}'...")
            index.storage_context.persist(persist_dir=persist_dir)
            logging.info("Index persisted.")
        else:
            logging.info(f"Loading existing index from '{persist_dir}'...")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            logging.info("Index loaded successfully.")
        return index

index = create_or_load_index(documents, PERSIST_DIR)

# Create query engine
def create_query_engine(index):
    if not index:
        logging.error("Error: 'index' variable not found.")
        raise SystemExit("Index creation/loading failed or skipped.")
    else:
        logging.info("Creating query engine...")
        query_engine = index.as_query_engine()
        logging.info("Query engine created.")
        return query_engine

query_engine = create_query_engine(index)

# Sample query
def run_sample_query(query_engine):
    sample_query = "What is the main topic discussed in the documents? Be detailed"
    logging.info(f"\n--- Running Sample Query ---")
    logging.info(f"Query: {sample_query}")
    response = query_engine.query(sample_query)
    logging.info("\nResponse:")
    logging.info(response)
    logging.info("--- End Sample Query ---")

run_sample_query(query_engine)

# Create chat engine
def create_chat_engine(index):
    if not index:
        logging.error("Error: 'index' variable not found.")
        raise SystemExit("Index creation/loading failed or skipped.")
    else:
        logging.info("Creating chat engine...")
        chat_engine = index.as_chat_engine(
            chat_mode='context',
            verbose=True
        )
        logging.info("Chat engine created. Type 'quit' or 'exit' to end the chat.")
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    logging.info("Bot: Goodbye!")
                    break
                response = chat_engine.chat(user_input)
                logging.info(f"Bot: {response}")
            except EOFError:
                logging.info("\nBot: Chat interrupted. Goodbye!")
                break
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                break

create_chat_engine(index)