"""
Memory — bộ nhớ ngắn hạn (list) + dài hạn (ChromaDB).
Hỗ trợ namespace để mỗi SwarmAgent có collection riêng.
"""
import chromadb
import json


class Memory:
    """
    namespace: tên collection ChromaDB riêng cho mỗi agent.
    Nếu namespace=None → dùng collection mặc định "agent_memory".
    """

    def __init__(self, namespace=None):
        self.short_term = []
        self.namespace = namespace or "agent_memory"
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.client.get_or_create_collection(name=self.namespace)
        except Exception as e:
            self.collection = None
            print(f"Warning: ChromaDB init failed (ns={self.namespace}):", e)

    def add_short_term(self, role, content):
        self.short_term.append({"role": role, "content": content})

    def get_short_term_context(self):
        return json.dumps(self.short_term, ensure_ascii=False)

    def save_long_term(self, task_id, desc, result):
        if not self.collection:
            return
        doc_id = f"task_{task_id}_{len(self.short_term)}"
        self.collection.add(
            documents=[f"Task: {desc} | Result: {result}"],
            metadatas=[{"task_id": task_id}],
            ids=[doc_id],
        )

    def search_similar(self, query, n=3):
        if not self.collection:
            return []
        try:
            results = self.collection.query(query_texts=[query], n_results=n)
            if results and "documents" in results and results["documents"]:
                return results["documents"][0]
            return []
        except Exception:
            return []


# Instance mặc định dùng cho chế độ đơn agent (backward compat)
memory = Memory()
