import os
import time
import pandas as pd
from dotenv import load_dotenv
import logging
from logging_config import setup_logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.utilities import SQLDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter

setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()

DB_URI = os.getenv("DB_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

VECTOR_INDEX_PATH = "vector_index"
TIMESTAMP_FILE = "vector_last_update.txt"

BATCH_SIZE = 500
BATCH_SLEEP = 15

_vector_db_instance = None


# Metadata Timestamp helpers
def get_last_update_time():
    if not os.path.exists(TIMESTAMP_FILE):
        return None
    with open(TIMESTAMP_FILE, "r") as f:
        return f.read().strip()


def set_last_update_time(ts):
    with open(TIMESTAMP_FILE, "w") as f:
        f.write(str(ts))


# Helper
def clean_location(value):
    if pd.isna(value):
        return None
    return str(value).replace("_", " ").strip()


# Fetch documents
def fetch_documents(since=None):

    db = SQLDatabase.from_uri(DB_URI)
    docs = []

    # COOPERATIVES

    query = """
    SELECT cooperative_id, cooperative_name, cooperative_type,
           cooperative_state, cooperative_county,
           approval_status, cooperative_date_created
    FROM cooperative
    """

    if since:
        query += f" WHERE cooperative_date_created > '{since}'"

    coop = pd.read_sql(query, db._engine)

    for _, r in coop.iterrows():

        text = f"""
        Cooperative {r.cooperative_name} is a {r.cooperative_type}
        cooperative in {clean_location(r.cooperative_county)},
        {clean_location(r.cooperative_state)}.
        Approval status: {r.approval_status}
        """

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "table": "cooperative",
                    "cooperative_id": r.cooperative_id
                }
            )
        )

    # MEMBERS
    query = """
    SELECT member_id, cooperative_id, member_name,
           member_gender, member_age, member_state,
           member_date_created
    FROM member
    """

    if since:
        query += f" WHERE member_date_created > '{since}'"

    members = pd.read_sql(query, db._engine)

    for _, r in members.iterrows():

        text = f"""
        Member {r.member_name} is {r.member_gender}
        aged {r.member_age} from {clean_location(r.member_state)}
        """

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "table": "member",
                    "member_id": r.member_id,
                    "cooperative_id": r.cooperative_id
                }
            )
        )

    # DIRECTORS
    query = """
    SELECT director_id, cooperative_id,
           director_name, director_gender,
           director_type, director_state,
           director_date_created
    FROM director
    """

    if since:
        query += f" WHERE director_date_created > '{since}'"

    directors = pd.read_sql(query, db._engine)

    for _, r in directors.iterrows():

        text = f"""
        Director {r.director_name} is a {r.director_type}
        director from {clean_location(r.director_state)}
        """

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "table": "director",
                    "director_id": r.director_id,
                    "cooperative_id": r.cooperative_id
                }
            )
        )

    # DEREGISTRATION
    query = """
    SELECT coop_id, reason, status, date_created
    FROM deregistration
    """

    if since:
        query += f" WHERE date_created > '{since}'"

    dereg = pd.read_sql(query, db._engine)

    for _, r in dereg.iterrows():

        text = f"""
        Cooperative deregistration reason {r.reason}
        with status {r.status}
        """

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "table": "deregistration",
                    "cooperative_id": r.coop_id
                }
            )
        )

    return docs


# Split documents
def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(docs)


# Build full index
def build_vector_db():

    global _vector_db_instance

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    if os.path.exists(VECTOR_INDEX_PATH):

        print("Loading existing FAISS index...")

        _vector_db_instance = FAISS.load_local(
            VECTOR_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return _vector_db_instance

    print("Creating FAISS index from scratch...")

    docs = fetch_documents()

    splits = split_documents(docs)

    total_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(splits), BATCH_SIZE):

        batch = splits[i:i+BATCH_SIZE]

        print(f"Embedding batch {i//BATCH_SIZE+1}/{total_batches}")

        index = FAISS.from_documents(batch, embeddings)

        if _vector_db_instance is None:
            _vector_db_instance = index
        else:
            _vector_db_instance.merge_from(index)

        if i+BATCH_SIZE < len(splits):
            time.sleep(BATCH_SLEEP)

    _vector_db_instance.save_local(VECTOR_INDEX_PATH)

    set_last_update_time(pd.Timestamp.now())

    print("Vector index created.")

    return _vector_db_instance


# Incremental update
# def update_vector_index():

#     global _vector_db_instance

#     if _vector_db_instance is None:
#         _vector_db_instance = build_vector_db()

#     last_time = get_last_update_time()

#     print("Checking for new database rows...")

#     docs = fetch_documents(last_time)

#     if not docs:
#         print("No new documents.")
#         return _vector_db_instance

#     splits = split_documents(docs)

#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="gemini-embedding-001",
#         google_api_key=GOOGLE_API_KEY
#     )

#     new_index = FAISS.from_documents(splits, embeddings)

#     _vector_db_instance.merge_from(new_index)

#     _vector_db_instance.save_local(VECTOR_INDEX_PATH)

#     set_last_update_time(pd.Timestamp.now())

#     print("Vector index updated.")

#     return _vector_db_instance

# vector_db.py (refactored update_vector_index)
def update_vector_index():
    """
    Incrementally update the FAISS vector index with new database rows.
    If no new data exists, reuse the saved index to avoid unnecessary API calls.
    """

    global _vector_db_instance

    # Step 1: Load existing index if not already loaded
    if _vector_db_instance is None:
        if os.path.exists(VECTOR_INDEX_PATH):
            logger.info("Loading existing FAISS index...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            _vector_db_instance = FAISS.load_local(
                VECTOR_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Existing vector index loaded.")
        else:
            logger.info("No existing index found. Building from scratch...")
            return build_vector_db()

    # Step 2: Check for new documents
    last_time = get_last_update_time()
    docs = fetch_documents(since=last_time)

    if not docs:
        logger.info("No new documents found. Using existing vector index.")
        return _vector_db_instance

    logger.info(f"Found {len(docs)} new documents. Updating vector index...")

    # Step 3: Split documents
    splits = split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Step 4: Embed in batches to avoid rate limits
    total_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(splits), BATCH_SIZE):
        batch = splits[i:i + BATCH_SIZE]
        logger.info(f"Embedding batch {i // BATCH_SIZE + 1}/{total_batches}...")
        new_index = FAISS.from_documents(batch, embeddings)

        _vector_db_instance.merge_from(new_index)

        if i + BATCH_SIZE < len(splits):
            logger.info(f"Sleeping for {BATCH_SLEEP}s to respect API rate limits...")
            time.sleep(BATCH_SLEEP)

    # Step 5: Save updated index
    _vector_db_instance.save_local(VECTOR_INDEX_PATH)
    set_last_update_time(pd.Timestamp.now())
    logger.info("Vector index successfully updated.")

    return _vector_db_instance

# Public getter
def get_vector_db():

    global _vector_db_instance

    if _vector_db_instance is None:
        logger.info("Loading FAISS vector database...")
        _vector_db_instance = build_vector_db()
        logger.info("Vector DB loaded successfully.")

    return _vector_db_instance