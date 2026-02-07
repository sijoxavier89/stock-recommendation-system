"""Placeholder vector store abstraction (e.g., Chroma wrapper)."""

_store = {}

def add(ticker: str, vector):
    _store.setdefault(ticker, []).append(vector)

def query(query_vector, top_k=5):
    return []
