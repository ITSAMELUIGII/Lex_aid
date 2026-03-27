import os
import re
import uuid
import json
import urllib.request
import ssl
import pickle
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Set up paths
DATA_DIR = "./data"
DB_DIR = "./chroma_db"
BM25_PATH = "./bm25_index.pkl"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# URLs of the legal acts
ACTS = {
    "CPA_2019": "https://ncdrc.nic.in/bare_acts/CPA2019.pdf",
    "MTA_2019": "https://mohua.gov.in/upload/whatsnew/5d25fb70671ebdraft%20Model%20Tenancy%20Act,%202019.pdf"
}

def download_pdfs():
    print("Downloading PDFs...")
    pdf_paths = []
    for name, url in ACTS.items():
        filepath = os.path.join(DATA_DIR, f"{name}.pdf")
        if not os.path.exists(filepath):
            # Adding User-Agent and bypassing SSL
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, context=ctx) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Downloaded {name}")
        else:
            print(f"{name} already downloaded")
        pdf_paths.append((name, filepath))
    return pdf_paths

def extract_text(pdf_paths):
    docs_text = {}
    for name, path in pdf_paths:
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            docs_text[name] = text
        except Exception as e:
            print(f"Error reading {name}: {e}")
    return docs_text

def split_by_section(text, act_name):
    # This regex attempts to find "CHAPTER [Roman/Number]" or "Section \d+" or just \d+\. as headers
    # A robust split for legal docs often requires custom regex. 
    # Let's split using a regex that looks for typical section headings.
    # E.g., "(?i)(?:\n\s*CHAPTER\s+[IVXLCDMC]+|\n\s*CHAPTER\s+\d+|\n\s*(?:Section|Article|Clause)?\s*\d+\.\s+)"
    
    # Simple split by newline + numbers followed by dot (e.g. " 1. ", " 2. ") or "Section X"
    # To keep it generic but effective, we'll use Langchain's RecursiveCharacterTextSplitter 
    # with custom separators prioritizing "CHAPTER ", "Section ", and "\n\d+\."
    
    separators = [
        "\nCHAPTER ",
        "\nSection ",
        "\nARTICLE ",
        r"\n\d+\.",
        "\n\n",
        "\n",
        ". "
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=1500,
        chunk_overlap=150,
        is_separator_regex=True
    )
    
    chunks = text_splitter.split_text(text)
    
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={"act": act_name, "chunk_index": i}
        )
        documents.append(doc)
    return documents

def main():
    pdf_paths = download_pdfs()
    raw_texts = extract_text(pdf_paths)
    
    all_documents = []
    for name, text in raw_texts.items():
        print(f"Splitting text for {name}...")
        docs = split_by_section(text, name)
        all_documents.extend(docs)
        print(f"Total chunks for {name}: {len(docs)}")
        
    if not all_documents:
        print("No documents were processed!")
        return

    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building and saving ChromaDB for Semantic Search...")
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("ChromaDB saved correctly.")
    
    print("Building BM25 for Keyword Search...")
    tokenized_corpus = [doc.page_content.lower().split() for doc in all_documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save the BM25 model alongside documents to disk
    with open(BM25_PATH, 'wb') as f:
        pickle.dump({"bm25": bm25, "documents": all_documents}, f)
    
    print("\nIngestion complete! All data stored locally.")

if __name__ == "__main__":
    main()
