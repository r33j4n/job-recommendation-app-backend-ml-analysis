import os
import shutil
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

import get_embedding

VECTOR_DB_PATH = "database"

logging.basicConfig(level=logging.DEBUG)

def populate_dbcv(file_paths):
    logging.debug("Starting populate_dbcv")
    documents = load_documents(file_paths)
    chunks = split_documents(documents)
    return add_to_vector_db(chunks)

def load_documents(file_paths):
    logging.debug("Loading documents")
    documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    logging.debug(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents):
    logging.debug("Splitting documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(documents)
    logging.debug(f"Split into {len(documents)} chunks")
    return documents

def add_to_vector_db(chunks: list[Document]):
    logging.debug("Adding to vector DB")
    try:
        vector_db = Chroma(
            persist_directory=VECTOR_DB_PATH, embedding_function=get_embedding.get_embedding_function()
        )
        chunks_with_ids = calculate_chunk_ids(chunks)
        existing_items = vector_db.get(include=[])
        existing_ids = set(existing_items["ids"])
        add_db_message = f"Number of existing documents in DB: {len(existing_ids)}"

        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            added_db_message = f"Adding new documents: {len(new_chunks)}"
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            vector_db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            added_db_message = "No new documents to add"

        response_messages = [add_db_message, added_db_message]
        logging.debug("Successfully added to vector DB")
        return response_messages
    except Exception as e:
        logging.error(f"Error adding to vector DB: {e}")
        raise

def calculate_chunk_ids(chunks):
    logging.debug("Calculating chunk IDs")
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_vector_db():
    logging.debug("Clearing vector DB")
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
    logging.debug("Cleared vector DB")