# Research: Graph Representation in DuckDB for Read-Only Databases

**Updated:** Rev 3 with STRUCT arrays and weakref pattern research

## Context

This research covers best practices for representing hierarchical/graph data in DuckDB, specifically for a read-only database that would be completely reconstructed if relationships change. The use case is anatomic locations with containment hierarchies, part-of relationships, and laterality links.

## Key Findings

### 1. Adjacency List Model is Ideal for Read-Only Scenarios

The **Adjacency List Model** is the most straightforward approach for representing hierarchies in SQL:

- Each record points to its immediate parent in the same table
- Self-referencing design is intuitive and maintainable
- Works best when reading small parts of the tree rather than large sub-trees
- Pairs well with recursive CTEs for traversal

For our anatomic locations use case (2,901 entries with max ~20 levels of hierarchy), this is the optimal choice.

**Sources:**
- [Storing Hierarchical Data in Relational Databases](https://adamdjellouli.com/articles/databases_notes/03_sql/09_hierarchical_data)

### 2. Recursive CTEs: Standard Pattern

DuckDB supports recursive CTEs with the standard SQL syntax:

```sql
WITH RECURSIVE ancestors AS (
    -- Base case: start node
    SELECT id, parent_id, 1 AS depth
    FROM hierarchy WHERE id = ?

    UNION ALL

    -- Recursive case: follow parent chain
    SELECT h.id, h.parent_id, a.depth + 1
    FROM hierarchy h
    JOIN ancestors a ON h.id = a.parent_id
    WHERE a.depth < 20  -- Prevent infinite loops
)
SELECT * FROM ancestors;
```

**Key considerations:**
- Always include a depth limit to prevent infinite loops in case of data errors
- Use `UNION ALL` for performance (unless deduplication is needed)
- For bidirectional traversal, create an edge table with both directions

### 3. USING KEY: New Performance Optimization (DuckDB 1.3+)

DuckDB 1.3 introduces `USING KEY` for recursive CTEs that dramatically improves performance for graph algorithms:

```sql
WITH RECURSIVE path(here, there, len) USING KEY (here, there) AS (
    -- Base case
    SELECT src, dst, 1 FROM edges
    UNION
    -- Recursive with key-based updates
    SELECT e.src, p.there, 1 + p.len
    FROM path p
    JOIN edges e ON e.dst = p.here
    LEFT JOIN recurring.path AS rec ON (rec.here = e.src AND rec.there = p.there)
    WHERE 1 + p.len < COALESCE(rec.len, 999)
)
SELECT * FROM path;
```

**Benefits:**
- Treats intermediate results as dictionaries (updates by key instead of append-only)
- Reduces memory consumption from O(n²) to O(unique keys)
- Dramatic performance gains on larger graphs

**For our use case:** Since we're doing simple ancestor/descendant queries (not shortest path), standard recursive CTEs are sufficient. USING KEY would only benefit if we were computing complex graph metrics.

**Sources:**
- [USING KEY in Recursive CTEs – DuckDB](https://duckdb.org/2025/05/23/using-key)
- [How DuckDB is USING KEY to Unlock Recursive Query Performance](https://duckdb.org/science/bamberg-using-key-sigmod/)

### 4. DuckPGQ Extension: Overkill for Simple Hierarchies

DuckPGQ adds SQL/PGQ (Property Graph Query) support:

```sql
CREATE PROPERTY GRAPH anatomy_graph
  VERTEX TABLES (anatomic_locations)
  EDGE TABLES (
    anatomic_hierarchy SOURCE KEY (child_id) REFERENCES anatomic_locations (id)
                       DESTINATION KEY (parent_id) REFERENCES anatomic_locations (id)
  );

-- Query with visual syntax
FROM GRAPH_TABLE (anatomy_graph
  MATCH (child:anatomic_locations)-[rel:anatomic_hierarchy]->{1,5}(ancestor:anatomic_locations)
  COLUMNS (child.id, ancestor.id, ancestor.description)
);
```

**Pros:**
- More intuitive syntax for graph patterns
- Path-finding with variable length
- Outperforms Neo4j on certain pattern matching queries

**Cons:**
- Community extension (portability concerns for downloadable databases)
- Overkill for simple parent-child hierarchies
- Adds dependency complexity

**Decision:** Not needed for our use case. Standard recursive CTEs are simpler and sufficient.

**Sources:**
- [Uncovering Financial Crime with DuckDB and Graph Queries](https://duckdb.org/2025/10/22/duckdb-graph-queries-duckpgq)
- [DuckPGQ Community Extension](https://duckdb.org/community_extensions/extensions/duckpgq)

### 5. Edge Table Design Patterns

For bidirectional traversal (needed for contains/containedBy):

```sql
-- Option A: Single table with explicit relationship direction
CREATE TABLE anatomic_hierarchy (
    child_id VARCHAR NOT NULL,
    parent_id VARCHAR NOT NULL,
    relationship_type VARCHAR NOT NULL,  -- 'containedBy' or 'partOf'
    parent_display VARCHAR,              -- Denormalized for fast display
    PRIMARY KEY (child_id, relationship_type)
);

-- Option B: Separate inverse relationship table (more explicit)
CREATE TABLE anatomic_children (
    parent_id VARCHAR NOT NULL,
    child_id VARCHAR NOT NULL,
    relationship_type VARCHAR NOT NULL,  -- 'contains' or 'hasParts'
    child_display VARCHAR,
    PRIMARY KEY (parent_id, child_id, relationship_type)
);
```

For our read-only use case, **Option A + B is best**: store both directions explicitly since the database is rebuilt from source anyway. This avoids runtime reversal and speeds up common queries.

**Sources:**
- [Graph Components with DuckDB](https://maxhalford.github.io/blog/graph-components-duckdb/)

### 6. Performance Considerations for Read-Only Databases

When the database is completely reconstructed on change:

1. **No need for foreign keys** - integrity is guaranteed by the build process
2. **Denormalize aggressively** - store display text alongside IDs for fast lookup
3. **Build all indexes after data load** - faster than maintaining during inserts
4. **Pre-compute common traversals** - for very hot paths, consider materialized views

### 7. Memory Scaling

Standard recursive CTEs have **quadratic memory scaling** for connected components:

| Component Size | Intermediate Rows |
|----------------|------------------|
| 100 nodes      | 10,000 rows      |
| 1,000 nodes    | 1,000,000 rows   |
| 10,000 nodes   | 100,000,000 rows |

For our anatomic locations:
- 2,901 entries total
- Largest connected component (Body region): ~25 entries max
- Deepest hierarchy: ~15-20 levels

**Conclusion:** Memory is not a concern for our dataset size.

### 8. Index Strategy for Graph Tables

```sql
-- For hierarchy traversal
CREATE INDEX idx_hierarchy_parent ON anatomic_hierarchy(parent_id);
CREATE INDEX idx_hierarchy_child ON anatomic_hierarchy(child_id);

-- For relationship-specific queries
CREATE INDEX idx_hierarchy_type ON anatomic_hierarchy(relationship_type);

-- Composite for common patterns
CREATE INDEX idx_hierarchy_lookup ON anatomic_hierarchy(child_id, relationship_type);
```

### 9. STRUCT Arrays vs JSON vs Parallel Arrays

For storing child references with both ID and display name, three approaches were evaluated:

| Approach | Schema | Performance | Complexity |
|----------|--------|-------------|------------|
| JSON string | `VARCHAR` | Parsing overhead | Medium |
| Parallel arrays | `ids VARCHAR[]`, `displays VARCHAR[]` | Native ops | Sync required |
| **STRUCT array** | `STRUCT(id, display)[]` | Native, vectorized | Low |

**Recommendation: STRUCT[]** - Native DuckDB type that leverages vectorized execution, maintains schema integrity, and avoids JSON parsing overhead.

```sql
containment_children STRUCT(id VARCHAR, display VARCHAR)[]
```

**Sources:**
- [DuckDB STRUCT: Handling Nested Data](https://motherduck.com/learn-more/duckdb-struct-nested-data/)
- [Shredding Deeply Nested JSON – DuckDB](https://duckdb.org/2023/03/03/json)

### 10. Weakref Pattern for Parent References in Pydantic

When child objects (AnatomicLocation) hold references to parent objects (AnatomicLocationIndex), circular references can cause memory leaks.

**The Problem:**
```
Index → returns Location → _index points back to Index
```

**The Solution:** Use `weakref.ref()` for the child→parent reference:

```python
import weakref
from pydantic import PrivateAttr, ConfigDict

class AnatomicLocation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _index: weakref.ReferenceType["AnatomicLocationIndex"] | None = PrivateAttr(default=None)

    def bind(self, index):
        self._index = weakref.ref(index)
        return self
```

**Benefits:**
- Breaks circular reference cycle
- Allows immediate garbage collection
- Enforces correct lifecycle (location fails if used outside index context)

**Sources:**
- [Circular References in Python](https://medium.com/@chipiga86/circular-references-without-memory-leaks-and-destruction-of-objects-in-python-43da57915b8d)
- [Pydantic weakref discussion](https://github.com/pydantic/pydantic/discussions/2857)

## Recommendations for Anatomic Locations

1. **Use materialized path pattern** - pre-compute hierarchy for instant queries
2. **Store children as STRUCT[]** - native DuckDB type, not JSON or parallel arrays
3. **Use weakref for index binding** - avoid circular reference memory leaks
4. **Skip DuckPGQ** - unnecessary complexity for simple hierarchies
5. **Denormalize display text** to avoid JOINs in common queries
6. **Drop/rebuild indexes** during database reconstruction
7. **No foreign keys** - simpler and matches existing index.py pattern
8. **Enforce context lifecycle** - locations must be used within index context

## References

1. [USING KEY in Recursive CTEs – DuckDB](https://duckdb.org/2025/05/23/using-key)
2. [How DuckDB is USING KEY to Unlock Recursive Query Performance](https://duckdb.org/science/bamberg-using-key-sigmod/)
3. [Uncovering Financial Crime with DuckDB and Graph Queries](https://duckdb.org/2025/10/22/duckdb-graph-queries-duckpgq)
4. [Graph Components with DuckDB](https://maxhalford.github.io/blog/graph-components-duckdb/)
5. [Storing Hierarchical Data in Relational Databases](https://adamdjellouli.com/articles/databases_notes/03_sql/09_hierarchical_data)
6. [DuckPGQ Community Extension](https://duckdb.org/community_extensions/extensions/duckpgq)
