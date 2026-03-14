from memory.semantic_memory import SemanticMemory

def test_semantic_retrieval():

    mem = SemanticMemory()

    queries = [
        "Why do ingestion failures happen?",
        "What causes preprocessing failures?",
        "Why does DART generation slow down?"
    ]

    for q in queries:
        print("\nQUERY:", q)
        print("RETRIEVED KNOWLEDGE:")
        print(mem.retrieve(q))
        print("-" * 50)


if __name__ == "__main__":
    test_semantic_retrieval()