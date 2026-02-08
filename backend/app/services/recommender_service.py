"""Recommender / RAG service.

Embeds a user query, retrieves relevant chunks from the vector store, assembles
context and calls the LLM to produce a structured answer. This implementation
is intentionally lightweight and defensive: if heavy dependencies (embeddings
or LLM) are not available the function returns a helpful error message.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

# Use relative imports within the services package so this module works when the
# package is imported as `backend.app.services` or as a module during tests.
from . import embedding_service, vector_store, llm_service

logger = logging.getLogger(__name__)


def _make_years(years: Optional[List[int]], last_n: int) -> List[int]:
	if years:
		return years
	current = datetime.now().year
	# default to last_n years ending with previous year
	return [current - i - 1 for i in range(last_n)][::-1]


def get_recommendation(ticker: str, user_query: str, years: Optional[List[int]] = None, last_n: int = 3, top_k: int = 5) -> Dict[str, Any]:
	"""Run a simple RAG pipeline and return LLM answer plus sources.

	Raises RuntimeError when required services aren't available.
	"""
	if not ticker or not user_query:
		raise ValueError("ticker and user_query are required")

	ticker = ticker.strip().upper()
	year_list = _make_years(years, last_n)

	if embedding_service is None:
		raise RuntimeError("Embedding service unavailable — install sentence-transformers or configure embedding provider")

	# 1) embed query
	try:
		query_emb = embedding_service.embed_text(user_query)
	except Exception as e:
		logger.error("Failed to embed query: %s", e)
		raise RuntimeError("Failed to compute query embedding")

	# 2) retrieve from vector store across years
	retrieved = []
	for y in year_list:
		try:
			res = vector_store.search_by_company(query_embedding=query_emb, ticker=ticker, year=y, n_results=top_k)
		except Exception as e:
			logger.warning("Vector store search failed for %s %s: %s", ticker, y, e)
			continue

		docs = res.get("documents", [])
		metas = res.get("metadatas", [])
		dists = res.get("distances", [])
		ids = res.get("ids", [])

		for i, doc in enumerate(docs):
			retrieved.append({
				"year": y,
				"text": doc,
				"metadata": metas[i] if i < len(metas) else {},
				"score": dists[i] if i < len(dists) else None,
				"id": ids[i] if i < len(ids) else None,
			})

	# sort by score (assume lower=better distance); if score missing, leave order
	try:
		retrieved = sorted([r for r in retrieved if r.get("text")], key=lambda r: (r.get("score") is None, r.get("score")))
	except Exception:
		pass

	# assemble context (limit tokens/chunks)
	MAX_CHUNKS = 12
	top_chunks = retrieved[:MAX_CHUNKS]

	context_parts = []
	sources = set()
	for idx, c in enumerate(top_chunks, start=1):
		md = c.get("metadata", {}) or {}
		src = md.get("file_path") or f"{ticker}_{c.get('year')}"
		if src:
			sources.add(str(src))
		header = f"[CHUNK {idx}] source={src} year={c.get('year')}"
		context_parts.append(header + "\n" + c.get("text", ""))

	context = "\n---\n".join(context_parts)

	system_prompt = (
		"You are a helpful financial analyst assistant. Answer briefly and cite sources. "
		"Use only the retrieved context unless you explicitly state otherwise."
	)

	user_instr = f"User question: {user_query}\n\nProvide a short structured answer and list sources."

	full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context}\n\n{user_instr}"

	try:
		llm_out = llm_service.generate(full_prompt)
	except Exception as e:
		logger.error("LLM generation failed: %s", e)
		# fallback: return retrieved snippets
		snippets = "\n\n".join([c["text"][:500] for c in top_chunks[:3]])
		llm_out = f"(LLM unavailable) Context snippets:\n{snippets}"

	return {
		"ticker": ticker,
		"years_searched": year_list,
		"retrieved_count": len(retrieved),
		"sources": list(sources),
		"llm_answer": llm_out,
		"retrieved_chunks": [{"id": c.get("id"), "year": c.get("year"), "text_snippet": (c.get("text") or "")[:400], "score": c.get("score")} for c in top_chunks],
	}
"""Recommender / RAG service.

Embeds a user query, retrieves relevant chunks from the vector store, assembles
context and calls the LLM to produce a structured answer. This implementation
is intentionally lightweight and defensive: if heavy dependencies (embeddings
or LLM) are not available the function returns a helpful error message.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

# Use relative imports within the services package so this module works when the
# package is imported as `backend.app.services` or as a module during tests.
from . import embedding_service, vector_store, llm_service

logger = logging.getLogger(__name__)


def _make_years(years: Optional[List[int]], last_n: int) -> List[int]:
	if years:
		return years
	current = datetime.now().year
	# default to last_n years ending with previous year
	return [current - i - 1 for i in range(last_n)][::-1]


def get_recommendation(ticker: str, user_query: str, years: Optional[List[int]] = None, last_n: int = 3, top_k: int = 5) -> Dict[str, Any]:
	"""Run a simple RAG pipeline and return LLM answer plus sources.

	Raises RuntimeError when required services aren't available.
	"""
	if not ticker or not user_query:
		raise ValueError("ticker and user_query are required")

	ticker = ticker.strip().upper()
	year_list = _make_years(years, last_n)

	if embedding_service is None:
		raise RuntimeError("Embedding service unavailable — install sentence-transformers or configure embedding provider")

	# 1) embed query
	try:
		query_emb = embedding_service.embed_text(user_query)
	except Exception as e:
		logger.error("Failed to embed query: %s", e)
		raise RuntimeError("Failed to compute query embedding")

	# 2) retrieve from vector store across years
	retrieved = []
	for y in year_list:
		try:
			res = vector_store.search_by_company(query_embedding=query_emb, ticker=ticker, year=y, n_results=top_k)
		except Exception as e:
			logger.warning("Vector store search failed for %s %s: %s", ticker, y, e)
			continue

		docs = res.get("documents", [])
		metas = res.get("metadatas", [])
		dists = res.get("distances", [])
		ids = res.get("ids", [])

		for i, doc in enumerate(docs):
			retrieved.append({
				"year": y,
				"text": doc,
				"metadata": metas[i] if i < len(metas) else {},
				"score": dists[i] if i < len(dists) else None,
				"id": ids[i] if i < len(ids) else None,
			})

	# sort by score (assume lower=better distance); if score missing, leave order
	try:
		retrieved = sorted([r for r in retrieved if r.get("text")], key=lambda r: (r.get("score") is None, r.get("score")))
	except Exception:
		pass

	# assemble context (limit tokens/chunks)
	MAX_CHUNKS = 12
	top_chunks = retrieved[:MAX_CHUNKS]

	context_parts = []
	sources = set()
	for idx, c in enumerate(top_chunks, start=1):
		md = c.get("metadata", {}) or {}
		src = md.get("file_path") or f"{ticker}_{c.get('year')}"
		if src:
			sources.add(str(src))
		header = f"[CHUNK {idx}] source={src} year={c.get('year')}"
		context_parts.append(header + "\n" + c.get("text", ""))

	context = "\n---\n".join(context_parts)

	system_prompt = (
		"You are a helpful financial analyst assistant. Answer briefly and cite sources. "
		"Use only the retrieved context unless you explicitly state otherwise."
	)

	user_instr = f"User question: {user_query}\n\nProvide a short structured answer and list sources."

	full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context}\n\n{user_instr}"

	try:
		llm_out = llm_service.generate(full_prompt)
	except Exception as e:
		logger.error("LLM generation failed: %s", e)
		# fallback: return retrieved snippets
		snippets = "\n\n".join([c["text"][:500] for c in top_chunks[:3]])
		llm_out = f"(LLM unavailable) Context snippets:\n{snippets}"

	return {
		"ticker": ticker,
		"years_searched": year_list,
		"retrieved_count": len(retrieved),
		"sources": list(sources),
		"llm_answer": llm_out,
		"retrieved_chunks": [{"id": c.get("id"), "year": c.get("year"), "text_snippet": (c.get("text") or "")[:400], "score": c.get("score")} for c in top_chunks],
	}

