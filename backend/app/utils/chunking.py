"""Text chunking utilities (placeholder)."""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    # naive split by characters for placeholder
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += chunk_size - overlap
    return chunks
