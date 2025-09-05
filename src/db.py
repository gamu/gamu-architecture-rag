from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
# Resolve paths relative to this file to avoid CWD issues
DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")
DB_TYPE = "faiss"
DOCS_DIRECTORY = os.path.join(os.path.dirname(__file__), "knowledge_base", "origin")

embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    add_start_index=True,
)

class DB:
    def __init__(self):
        self.k = 1

    def process_documents(self, directory_path):
        documents = []
    
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(('.txt')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    
                        metadata = {
                            "source": file_path,
                            "title": os.path.splitext(file)[0]
                        }
                    
                        chunks = text_splitter.create_documents([text], [metadata])
                    
                        for i, chunk in enumerate(chunks):
                            chunk.metadata["chunk_id"] = i
                            chunk.metadata["start_index"] = chunk.metadata.get("start_index", 0)
                    
                        documents.extend(chunks)
                
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
    
        return documents
    
    def create_vector_db(self, documents):

        db = FAISS.from_documents(documents, embeddings)
        db.save_local(DB_DIR)

    def create(self):
        documents =  self.process_documents(DOCS_DIRECTORY)
        print(f"Created {len(documents)} chunks")
    
        self.create_vector_db(documents)
        print(f"DB created in  {DB_DIR}")

