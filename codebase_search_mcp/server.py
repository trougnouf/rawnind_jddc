#!/usr/bin/env python3
"""MCP server for semantic codebase search via Roo-Code indexed collections."""

import os
from typing import TypedDict
import httpx
from mcp import FastMCP

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.woodpecker-garibaldi.ts.net:443")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://sigzil.woodpecker-garibaldi.ts.net:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "hf.co/nomic-ai/nomic-embed-code-GGUF:Q6_K")

mcp = FastMCP("codebase-search")


class SearchResult(TypedDict):
    """Single code search result with location and content."""
    file_path: str
    line: int
    score: float
    snippet: str
    additional_lines: list[int]


class CollectionInfo(TypedDict):
    """Qdrant collection metadata."""
    name: str
    vectors_count: int
    points_count: int


async def list_collections() -> list[str]:
    """List all available Qdrant collections."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{QDRANT_URL}/collections")
        resp.raise_for_status()
        collections = resp.json().get("result", {}).get("collections", [])
        return [c["name"] for c in collections]


async def generate_embedding(text: str) -> list[float]:
    """Generate embedding for query text via Ollama."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text}
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]


async def search_qdrant(collection: str, query_vector: list[float], limit: int) -> dict:
    """Search Qdrant collection with query vector."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{QDRANT_URL}/collections/{collection}/points/search",
            json={
                "vector": query_vector,
                "limit": limit,
                "with_payload": True,
                "with_vector": False
            }
        )
        resp.raise_for_status()
        return resp.json()


def parse_results(results: dict) -> list[SearchResult]:
    """Parse Qdrant results into structured SearchResult objects."""
    points = results.get("result", [])

    # Group by file to deduplicate
    by_file: dict[str, list[dict]] = {}
    for point in points:
        payload = point.get("payload", {})
        file_path = payload.get("file", payload.get("path", "unknown"))

        if file_path not in by_file:
            by_file[file_path] = []

        line_num = payload.get("line", payload.get("start_line", 0))
        # Convert to int if possible
        try:
            line_num = int(line_num) if line_num != "?" else 0
        except (ValueError, TypeError):
            line_num = 0

        by_file[file_path].append({
            "score": float(point.get("score", 0.0)),
            "line": line_num,
            "text": payload.get("text", payload.get("content", ""))
        })

    # Build structured results
    structured: list[SearchResult] = []
    for file_path, chunks in sorted(by_file.items()):
        # Take top chunk per file
        top_chunk = max(chunks, key=lambda x: x["score"])
        other_lines = [c["line"] for c in chunks[1:] if c["line"] != top_chunk["line"]]

        structured.append({
            "file_path": file_path,
            "line": top_chunk["line"],
            "score": top_chunk["score"],
            "snippet": top_chunk["text"][:200],
            "additional_lines": other_lines[:5]  # Limit to 5 additional
        })

    return structured


@mcp.tool()
async def get_collections() -> list[str]:
    """List all available Qdrant collections for code search.

    Returns:
        List of collection names (e.g., ['ws-abc123', 'ws-def456'])
    """
    return await list_collections()


@mcp.tool()
async def search_codebase(query: str, collection: str, limit: int = 10) -> list[SearchResult]:
    """Semantic search across indexed codebase using Roo-Code vector index.

    Args:
        query: Semantic search query (e.g., 'crop validation logic', 'alignment calculations')
        collection: Qdrant collection name. Use get_collections() to list available collections.
        limit: Max results to return (default: 10, max: 50)

    Returns:
        List of search results with file paths, line numbers, scores, and code snippets.
        Results are deduplicated per file (top match per file with additional line numbers).
    """
    # Validate limit
    limit = min(max(1, limit), 50)

    # Generate query embedding
    query_vector = await generate_embedding(query)

    # Search Qdrant
    results = await search_qdrant(collection, query_vector, limit)

    # Parse to structured format
    return parse_results(results)


if __name__ == "__main__":
    mcp.run()
