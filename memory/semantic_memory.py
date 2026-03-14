import chromadb
from sentence_transformers import SentenceTransformer


class SemanticMemory:

    def __init__(self):

        # persistent vector database
        self.client = chromadb.PersistentClient(
            path="memory/semantic_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="pipeline_knowledge"
        )

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")


    def add_knowledge(self, text, metadata=None):

        embedding = self.embedder.encode(text).tolist()

        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata if metadata else {"type": "semantic"}],
            ids=[str(hash(text))]
        )


    def retrieve(self, query: str, k: int = 3) -> str:

        embedding = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )

        if not results["documents"]:
            return ""

        docs = results["documents"][0]

        print("Semantic memory retrieved:", docs)

        return "\n".join(docs)