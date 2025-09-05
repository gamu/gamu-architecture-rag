from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import argparse
import os
import re

import db

from huggingface_hub import hf_hub_download

#model_path = hf_hub_download(
#    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
#    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#    local_dir="./models"
#)
#print(f"Model saved to: {model_path}")

class RAG:
    def __init__(
        self,
        db_type: str = "faiss",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "local",
        db_path: str = "vector_db",
        temperature: float = 0.7
    ):

        self.db_type = db_type
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.db_path = db_path
        self.temperature = temperature
    
        self._init_embeddings()
        self._load_vector_db()
        self._init_llm()
        
        self.cot_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a cautious, safety‑first RAG assistant. Answer ONLY based on the Context section below. "
             "Do not use prior knowledge, speculation, or assumptions. "
             "If information is insufficient or the question is out of scope, reply exactly: 'Insufficient data in context'. "
             "Never disclose passwords, emails, keys, or other sensitive data; if asked, reply: 'Request denied for security reasons'. "
             "Before the final answer, provide a brief Chain-of-Thought as 1–3 numbered steps. For example:\n"
             "1) First, find which technology is used in HyperRelay.\n"
             "2) The document states that HyperRelay is powered by the VoidCore engine.\n"
             "3) Therefore, the answer is VoidCore.\n\n"
             "Few-shot examples:\n"
             "Example 1\n"
             "Q: Who is mistakenly presumed dead and left behind?\n"
             "Context: Tammy Stein stars as astronaut, Sherri Green, who is mistakenly presumed dead and left behind on Aaron Rodriguez.\n"
             "Reasoning:\n"
             "1) The question asks who is presumed dead and left behind.\n"
             "2) The context explicitly names Sherri Green as that person.\n"
             "3) Therefore, the answer is Sherri Green.\n"
             "Final answer: Sherri Green.\n\n"
             "Example 2\n"
             "Q: How many sols was Watney alone on Aaron Rodriguez?\n"
             "Context: ... effectively rescuing him after being alone for 543 Sols on Aaron Rodriguez.\n"
             "Reasoning:\n"
             "1) The question asks for the number of sols.\n"
             "2) The context states the exact duration: 543 Sols.\n"
             "3) Therefore, the answer is 543 sols.\n"
             "Final answer: 543 sols."),
            ("human",
             "Question: {question}\n\nContext:\n{context}\n\n"
             "Briefly reason (1–3 points), then write:\n"
             "Final answer: <concise answer, 1–2 sentences>."
            )
        ])

        self._forbidden_patterns = [
            r"(?i)ignore\s+all\s+instructions",
        ]

        self.sensitive_patterns = [
            r'(?i)\s*пароль\s*',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        ]

    
    def _init_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
    
    def _load_vector_db(self):
       self.db = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
    
    def _init_llm(self):
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)

        default_repo = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
        default_file = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

        model_repo = os.environ.get("GGUF_REPO", default_repo)
        model_file = os.environ.get("GGUF_FILE", default_file)

        local_model_path = os.path.join(models_dir, model_file)

        if not os.path.exists(local_model_path):
            local_model_path = hf_hub_download(
                repo_id=model_repo,
                filename=model_file,
                local_dir=models_dir
            )

        self.llm = LlamaCpp(
            model_path=local_model_path,
            allow_dangerous_deserialization=True,
            temperature=self.temperature,
            n_ctx=2048
        )
    
    def retrieve_chunks(self, query: str, k: int = 5) -> List[Document]:
        docs =  self.db.similarity_search(query, k=k)
        safe_docs = self._filter_documents(docs)

        return safe_docs

    def _is_query_secure(self, query: str) -> bool:
        if len(query) > 1000:
            return False

        if not any(char.isalpha() for char in query):
            return True

        query_lower = query.lower()
        
        for pattern in self._forbidden_patterns:
            if re.search(pattern, query_lower):
                return False
            
        return True

    def _has_sensitive(self, text: str) -> bool:
        for pattern in self.sensitive_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _extract_final_answer(self, reasoning) -> str:
        text = None
        try:
            # Support AIMessage or objects with 'content'
            if hasattr(reasoning, 'content'):
                text = reasoning.content
            else:
                text = str(reasoning)
        except Exception:
            text = str(reasoning)

        for marker in ["Final answer:"]:
            if marker in text:
                return text.split(marker)[-1].strip()
        return text.strip()

    def _filter_documents(self, documents: List[Document]) -> List[Document]:
        return [doc for doc in documents if not self._has_sensitive(doc.page_content)]

    def ask(self, question: str, k: int = 3) -> Dict:
        if self._is_query_secure(question)==False:
            return {
                "answer": "Request rejected for security reasons",
                "sources": [],
                "is_rejected": True,
                "security_message": "Warning: the query is flagged as unsafe. Answer limited to refusal."
            }

        chunks = self.retrieve_chunks(question, k=k)

        context = "\n---\n".join(
            f"Source: {chunk.metadata['source']}\nText: {chunk.page_content}" 
            for chunk in chunks
        )

        cot_chain = self.cot_prompt | self.llm
        
        reasoning_process = cot_chain.invoke({
            "question": question,
            "context": context
        })
        
        final_answer = self._extract_final_answer(reasoning_process)
        
        return {
            "answer": final_answer,
            "reasoning": reasoning_process,
            "sources": list(set(chunk.metadata['source'] for chunk in chunks))
        }

class RAGREPL:
    def __init__(self, db_type="faiss", llm_provider="local", db_path="vector_db", k=3):
        self.rag = RAG(db_type=db_type, llm_provider=llm_provider, db_path=db_path)
        self.k = k
        self.history = []
        
    def print_help(self):
        print("\nCommands:")
        print("  /help - help")
        print("  /k <int> - Change chunks, default is 3")
        print("  /history - Show query history")
        print("  /source <number> - Answer source")
        print("  /exit - REPL exit")
        print("  <query> - Ask an question\n")
    
    def print_welcome(self):
        print("\nRAG REPL")
        print(f"Config: {self.rag.db_type} DB, {self.rag.llm_provider} LLM")
        print("Input an query or /help\n")
    
    def run(self):
        self.print_welcome()
        
        while True:
            try:
                user_input = input("rag> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.startswith('/'):
                    cmd = user_input.split()[0].lower()
                    
                    if cmd == '/help':
                        self.print_help()
                    
                    elif cmd == '/k' and len(user_input.split()) > 1:
                        try:
                            self.k = int(user_input.split()[1])
                            print(f"Chunks set to: {self.k}")
                        except ValueError:
                            print("Error: input integer number after /k")
                    
                    elif cmd == '/history':
                        for i, item in enumerate(self.history, 1):
                            print(f"{i}. {item['question']}")
                    
                    elif cmd == '/source' and len(user_input.split()) > 1:
                        try:
                            hist_num = int(user_input.split()[1]) - 1
                            if 0 <= hist_num < len(self.history):
                                self._print_source(self.history[hist_num])
                            else:
                                print("Wrong history entity number")
                        except ValueError:
                            print("Error: input integer number after /source")
                    
                    elif cmd in ('/exit', '/quit'):
                        print("exit...")
                        break
                    
                    else:
                        print("Unknown command, use /help")
                else:
                    result = self.rag.ask(user_input, k=self.k)
                    self.history.append({
                        "question": user_input,
                        "result": result
                    })
                    
                    if result.get("is_rejected"):
                        print("\n[SECURITY] Warning: the query is flagged as unsafe. Answer limited to refusal.")
                        if result.get("security_message"):
                            print(result["security_message"]) 

                    print("\nAnswer:")
                    print(result["answer"])
                    print("\nSource:")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"  {i}. {source}")
                    print()
            
            except KeyboardInterrupt:
                print("\nInput /exit for quit REPL")
                continue
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def _print_source(self, history_item):
        print(f"\nQuery: {history_item['question']}")
        print("\nFragments:")
        for i, chunk in enumerate(history_item['result']['chunks'], 1):
            print(f"\nFragment {i}:")
            print(f"Source: {chunk['source']}")
            print(f"Pos: {chunk['position']}")
            print("Text:")
            print(chunk['text'])
            print("-" * 50)

def main():
    dbMaker = db.DB()
    dbMaker.create()

    parser = argparse.ArgumentParser(description='RAG REPL Console')
    parser.add_argument('--db', type=str, default='faiss', 
                       help='DB type (faiss)')
    parser.add_argument('--llm', type=str, default='local', 
                       help='provider LLM (local)')
    parser.add_argument('--path', type=str, default='vector_db', 
                       help='DB path')
    parser.add_argument('--k', type=int, default=3, 
                       help='Chunks')
    
    args = parser.parse_args()
    
    repl = RAGREPL(db_type=args.db, llm_provider=args.llm, 
                  db_path=args.path, k=args.k)
    repl.run()


if __name__ == "__main__":
    main()