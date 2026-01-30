# Code Review Techniques

## Detecting Duplicate Code

### Use Serena's find_symbol (Preferred)

```
find_symbol("function_name", relative_path="packages")
```

- **Semantic**: Finds function definitions, not text matches
- **Shows signatures**: Immediately reveals implementation differences
- **Cross-package**: Searches entire codebase

### Systematic Check

For each function in a shared package (e.g., oidm-common), verify no duplicates exist elsewhere:

```
find_symbol("get_embedding", relative_path="packages")
find_symbol("get_embeddings_batch", relative_path="packages")
```

### Known Duplications Found (2025-01-18)

| Function | Locations | Should be in |
|----------|-----------|--------------|
| `get_embedding` | oidm-common, findingmodel-ai/_internal | oidm-common only |
| `get_embeddings_batch` | oidm-common, findingmodel-ai/_internal | oidm-common only |
| `strip_quotes` | findingmodel, findingmodel-ai | oidm-common |
| `strip_quotes_secret` | findingmodel, findingmodel-ai | oidm-common |

### Why grep is insufficient

- Finds text patterns, not semantic definitions
- Misses renamed functions with same logic
- No signature comparison
- False positives from comments/strings

### Questions to ask during review

1. "What shared utilities does oidm-common provide?"
2. "Does this package duplicate any oidm-common functionality?"
3. "For each function in _internal/, does it already exist elsewhere?"
