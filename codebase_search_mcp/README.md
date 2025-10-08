# Codebase Search MCP Server

Semantic code search using Roo-Code indexed collections in Qdrant.

## Installation

```bash
cd codebase_search_mcp
uv sync
```

This creates an isolated `.venv` with all dependencies.

## Configuration

Add to your MCP settings (e.g., `~/.claude/config.json`):

```json
{
  "mcpServers": {
    "codebase-search": {
      "command": "/home/rengo/rawnind_jddc/codebase_search_mcp/.venv/bin/python",
      "args": ["/home/rengo/rawnind_jddc/codebase_search_mcp/server.py"],
      "env": {
        "QDRANT_URL": "https://qdrant.woodpecker-garibaldi.ts.net:443",
        "OLLAMA_URL": "http://sigzil.woodpecker-garibaldi.ts.net:11434",
        "EMBEDDING_MODEL": "hf.co/nomic-ai/nomic-embed-code-GGUF:Q6_K"
      }
    }
  }
}
```

## Usage

```python
# List available collections
collections = get_collections()

# Search codebase
results = search_codebase("crop validation logic", "ws-abc123", limit=10)
# Returns list[SearchResult] with:
# - file_path: str
# - line: int
# - score: float (similarity 0-1)
# - snippet: str (code excerpt, max 200 chars)
# - additional_lines: list[int] (other matches in same file)
```

## Tools

- `get_collections()` - List all Qdrant collections
- `search_codebase(query, collection, limit)` - Semantic code search
