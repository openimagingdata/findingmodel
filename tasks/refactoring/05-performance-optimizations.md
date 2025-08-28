# Performance Optimizations Plan

## Executive Summary
The codebase has several performance bottlenecks including redundant database queries, inefficient batch operations, and missed opportunities for concurrent execution. This refactoring will optimize critical paths for 30-50% performance improvement.

**Impact**: Significant performance gains, reduced latency, better resource utilization
**Risk**: Low - Performance improvements are additive
**Effort**: 1 week for implementation and benchmarking

## Current Performance Analysis

### Identified Bottlenecks

#### 1. Redundant Database Queries
```python
# Current: Multiple get() calls in validation
async def _get_validation_data(self):
    # Makes 3+ separate queries
    for entry in entries:
        person = await self.get_person(entry.contributor)  # N queries
        org = await self.get_organization(person.org_code)  # N more queries
```

#### 2. Sequential Processing
```python
# Current: Process files one by one
async def update_from_directory(self, directory):
    for file in files:
        model = self.load_model(file)  # Sequential
        await self.validate_model(model)  # Sequential
        await self.save_model(model)  # Sequential
```

#### 3. Inefficient Batch Operations
```python
# Current: Individual inserts in batch operations
async def _execute_batch_operations(self, operations):
    for op in operations:
        if op.type == "insert":
            await self.collection.insert_one(op.document)  # N database calls
```

#### 4. Missing Caching
- File hash calculations repeated
- Validation data fetched multiple times
- No connection pooling
- No query result caching

## Target Performance Improvements

### Optimization Goals
- **Batch Operations**: 50% faster through proper batching
- **Validation**: 40% faster through caching and query optimization
- **File Processing**: 60% faster through concurrent I/O
- **Search**: 30% faster through better indexing and caching

## Implementation Details

### 1. Database Query Optimization

#### 1.1 Query Batching
```python
# src/findingmodel/index/optimized_repository.py
class OptimizedRepository:
    """Repository with query optimization."""
    
    def __init__(self, db, max_batch_size=1000):
        self.db = db
        self.max_batch_size = max_batch_size
    
    async def get_many_by_ids(
        self, 
        ids: list[str], 
        collection_name: str
    ) -> dict[str, dict]:
        """Fetch multiple documents in single query."""
        collection = self.db[collection_name]
        
        # Single query for all IDs
        cursor = collection.find({"_id": {"$in": ids}})
        results = {}
        async for doc in cursor:
            results[doc["_id"]] = doc
        
        return results
    
    async def bulk_insert(
        self, 
        documents: list[dict], 
        collection_name: str
    ) -> list[str]:
        """Insert multiple documents efficiently."""
        collection = self.db[collection_name]
        
        # Split into batches if needed
        inserted_ids = []
        for i in range(0, len(documents), self.max_batch_size):
            batch = documents[i:i + self.max_batch_size]
            result = await collection.insert_many(batch, ordered=False)
            inserted_ids.extend(result.inserted_ids)
        
        return inserted_ids
    
    async def bulk_update(
        self,
        updates: list[tuple[dict, dict]],  # [(filter, update), ...]
        collection_name: str
    ) -> int:
        """Update multiple documents efficiently."""
        collection = self.db[collection_name]
        
        # Use bulk_write for efficiency
        operations = [
            UpdateOne(filter_doc, update_doc, upsert=True)
            for filter_doc, update_doc in updates
        ]
        
        result = await collection.bulk_write(operations, ordered=False)
        return result.modified_count + result.upserted_count
```

#### 1.2 Query Result Caching
```python
# src/findingmodel/index/cache.py
from functools import lru_cache
from typing import Any, Optional
import asyncio
import time

class QueryCache:
    """LRU cache for query results."""
    
    def __init__(self, max_size=1000, ttl=300):  # 5 min TTL
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}
        self._lock = asyncio.Lock()
    
    def _make_key(self, query: dict, collection: str) -> str:
        """Create cache key from query."""
        import hashlib
        import json
        
        query_str = json.dumps(query, sort_keys=True)
        return f"{collection}:{hashlib.md5(query_str.encode()).hexdigest()}"
    
    async def get(
        self, 
        query: dict, 
        collection: str
    ) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._make_key(query, collection)
        
        async with self._lock:
            if key in self._cache:
                timestamp = self._timestamps[key]
                if time.time() - timestamp < self.ttl:
                    return self._cache[key]
                else:
                    # Expired
                    del self._cache[key]
                    del self._timestamps[key]
        
        return None
    
    async def set(
        self, 
        query: dict, 
        collection: str, 
        result: Any
    ):
        """Cache query result."""
        key = self._make_key(query, collection)
        
        async with self._lock:
            # Implement LRU eviction
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(
                    self._timestamps, 
                    key=self._timestamps.get
                )
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = result
            self._timestamps[key] = time.time()
    
    async def invalidate(self, collection: Optional[str] = None):
        """Invalidate cache entries."""
        async with self._lock:
            if collection:
                # Invalidate specific collection
                keys_to_remove = [
                    k for k in self._cache 
                    if k.startswith(f"{collection}:")
                ]
                for key in keys_to_remove:
                    del self._cache[key]
                    del self._timestamps[key]
            else:
                # Clear all
                self._cache.clear()
                self._timestamps.clear()
```

### 2. Concurrent Processing

#### 2.1 Batch Processor with Concurrency Control
```python
# src/findingmodel/index/batch_processor.py
import asyncio
from typing import TypeVar, Callable, Awaitable, List
from itertools import islice

T = TypeVar('T')
U = TypeVar('U')

class BatchProcessor:
    """Process items concurrently with rate limiting."""
    
    def __init__(
        self, 
        max_concurrency: int = 20,
        batch_size: int = 100,
        rate_limit: Optional[int] = None  # requests per second
    ):
        self.max_concurrency = max_concurrency
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.rate_limit = rate_limit
        self._last_request_time = 0
    
    async def _rate_limit_wait(self):
        """Wait if rate limiting is needed."""
        if self.rate_limit:
            min_interval = 1.0 / self.rate_limit
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_request_time = time.time()
    
    async def _process_with_limit(
        self, 
        processor: Callable[[T], Awaitable[U]], 
        item: T
    ) -> U:
        """Process single item with concurrency and rate limiting."""
        async with self.semaphore:
            await self._rate_limit_wait()
            return await processor(item)
    
    async def process_batch(
        self,
        items: List[T],
        processor: Callable[[T], Awaitable[U]],
        chunk_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[U]:
        """Process items in parallel batches."""
        chunk_size = chunk_size or self.batch_size
        results = []
        total = len(items)
        processed = 0
        
        # Process in chunks to manage memory
        for chunk in self._chunked(items, chunk_size):
            # Create tasks for chunk
            tasks = [
                self._process_with_limit(processor, item)
                for item in chunk
            ]
            
            # Wait for chunk completion
            chunk_results = await asyncio.gather(
                *tasks, 
                return_exceptions=True
            )
            
            # Handle results
            for result in chunk_results:
                if isinstance(result, Exception):
                    # Log error but continue
                    print(f"Error processing item: {result}")
                    results.append(None)
                else:
                    results.append(result)
            
            # Progress callback
            processed += len(chunk)
            if progress_callback:
                progress_callback(processed, total)
        
        return results
    
    @staticmethod
    def _chunked(iterable, size):
        """Split iterable into fixed-size chunks."""
        it = iter(iterable)
        while chunk := list(islice(it, size)):
            yield chunk

class StreamProcessor:
    """Process async iterators efficiently."""
    
    def __init__(self, max_buffer: int = 100):
        self.max_buffer = max_buffer
    
    async def map_stream(
        self,
        stream: AsyncIterator[T],
        processor: Callable[[T], Awaitable[U]],
        concurrency: int = 10
    ) -> AsyncIterator[U]:
        """Map processor over async stream with concurrency."""
        queue = asyncio.Queue(maxsize=self.max_buffer)
        
        async def producer():
            """Read from stream into queue."""
            async for item in stream:
                await queue.put(item)
            await queue.put(None)  # Sentinel
        
        async def consumer():
            """Process items from queue."""
            semaphore = asyncio.Semaphore(concurrency)
            tasks = []
            
            while True:
                item = await queue.get()
                if item is None:
                    break
                
                async with semaphore:
                    result = await processor(item)
                    yield result
        
        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        async for result in consumer():
            yield result
        
        await producer_task
```

#### 2.2 File Processing Optimization
```python
# src/findingmodel/index/file_processor.py
import aiofiles
import hashlib
from pathlib import Path

class OptimizedFileProcessor:
    """Optimized file operations."""
    
    def __init__(self, batch_processor: BatchProcessor):
        self.batch_processor = batch_processor
        self._hash_cache = {}
    
    async def calculate_hash_async(self, file_path: Path) -> str:
        """Calculate file hash asynchronously with caching."""
        # Check cache
        cache_key = (str(file_path), file_path.stat().st_mtime)
        if cache_key in self._hash_cache:
            return self._hash_cache[cache_key]
        
        # Calculate hash asynchronously
        sha256_hash = hashlib.sha256()
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        
        result = sha256_hash.hexdigest()
        self._hash_cache[cache_key] = result
        return result
    
    async def load_models_concurrent(
        self, 
        file_paths: List[Path]
    ) -> List[FindingModelFull]:
        """Load multiple model files concurrently."""
        
        async def load_single(path: Path) -> FindingModelFull:
            async with aiofiles.open(path, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                return FindingModelFull(**data)
        
        return await self.batch_processor.process_batch(
            file_paths,
            load_single,
            chunk_size=50
        )
    
    async def process_directory_optimized(
        self, 
        directory: Path,
        pattern: str = "*.fm.json"
    ) -> dict:
        """Process directory with optimized I/O."""
        # Find files
        files = sorted(directory.rglob(pattern))
        
        # Process files concurrently
        async def process_file(file_path: Path):
            # Calculate hash and load model in parallel
            hash_task = self.calculate_hash_async(file_path)
            model_task = self.load_models_concurrent([file_path])
            
            file_hash, models = await asyncio.gather(
                hash_task, model_task
            )
            
            return {
                'path': str(file_path),
                'hash': file_hash,
                'model': models[0] if models else None
            }
        
        results = await self.batch_processor.process_batch(
            files,
            process_file,
            chunk_size=20
        )
        
        return {r['path']: r for r in results if r}
```

### 3. Connection Pooling

```python
# src/findingmodel/index/connection_pool.py
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import asyncio

class MongoConnectionPool:
    """MongoDB connection pool manager."""
    
    _instance: Optional['MongoConnectionPool'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._pools = {}
            self._initialized = True
    
    async def get_client(
        self,
        connection_string: str,
        max_pool_size: int = 100,
        min_pool_size: int = 10
    ) -> AsyncIOMotorClient:
        """Get or create connection pool."""
        async with self._lock:
            if connection_string not in self._pools:
                # Create new connection pool
                self._pools[connection_string] = AsyncIOMotorClient(
                    connection_string,
                    maxPoolSize=max_pool_size,
                    minPoolSize=min_pool_size,
                    maxIdleTimeMS=30000,  # 30 seconds
                    waitQueueTimeoutMS=5000,  # 5 seconds
                    serverSelectionTimeoutMS=3000,  # 3 seconds
                    connectTimeoutMS=2000,  # 2 seconds
                    socketTimeoutMS=10000  # 10 seconds
                )
            
            return self._pools[connection_string]
    
    async def close_all(self):
        """Close all connection pools."""
        async with self._lock:
            for client in self._pools.values():
                client.close()
            self._pools.clear()
```

### 4. Search Optimization

```python
# src/findingmodel/index/optimized_search.py
from typing import List, Dict, Any
import asyncio

class OptimizedSearchEngine:
    """Search engine with performance optimizations."""
    
    def __init__(self, repository, cache: QueryCache):
        self.repository = repository
        self.cache = cache
        self._text_index_created = False
    
    async def ensure_text_index(self):
        """Ensure text index exists for search."""
        if not self._text_index_created:
            collection = self.repository.collection
            
            # Create compound text index
            await collection.create_index([
                ("name", "text"),
                ("description", "text"),
                ("synonyms", "text")
            ])
            
            # Create supporting indexes
            await collection.create_index([("oifm_id", 1)])
            await collection.create_index([("slug_name", 1)])
            
            self._text_index_created = True
    
    async def search_optimized(
        self,
        query: str,
        limit: int = 10,
        use_cache: bool = True
    ) -> List[IndexEntry]:
        """Optimized search with caching."""
        await self.ensure_text_index()
        
        # Check cache
        if use_cache:
            cache_key = {"query": query, "limit": limit}
            cached = await self.cache.get(cache_key, "search")
            if cached:
                return cached
        
        # Build optimized query
        search_filter = {
            "$or": [
                {"$text": {"$search": query}},
                {"oifm_id": query},
                {"slug_name": query.lower()}
            ]
        }
        
        # Use projection to reduce data transfer
        projection = {
            "oifm_id": 1,
            "name": 1,
            "description": 1,
            "score": {"$meta": "textScore"}
        }
        
        # Execute search with score sorting
        cursor = self.repository.collection.find(
            search_filter,
            projection
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        
        results = []
        async for doc in cursor:
            results.append(IndexEntry(**doc))
        
        # Cache results
        if use_cache:
            await self.cache.set(cache_key, "search", results)
        
        return results
    
    async def search_batch_optimized(
        self,
        queries: List[str],
        limit_per_query: int = 10
    ) -> Dict[str, List[IndexEntry]]:
        """Batch search with concurrent execution."""
        
        async def search_single(query: str):
            results = await self.search_optimized(query, limit_per_query)
            return (query, results)
        
        # Execute searches concurrently
        tasks = [search_single(q) for q in queries]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
```

### 5. Performance Monitoring

```python
# src/findingmodel/index/monitoring.py
import time
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List
import statistics

@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation: str
    duration_ms: float
    item_count: int = 1
    
    @property
    def throughput(self) -> float:
        """Items per second."""
        if self.duration_ms == 0:
            return 0
        return self.item_count / (self.duration_ms / 1000)

class PerformanceMonitor:
    """Monitor and report performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[PerformanceMetrics]] = {}
    
    @asynccontextmanager
    async def measure(self, operation: str, item_count: int = 1):
        """Context manager to measure operation performance."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            metric = PerformanceMetrics(
                operation=operation,
                duration_ms=duration_ms,
                item_count=item_count
            )
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(metric)
    
    def get_stats(self, operation: str) -> dict:
        """Get statistics for an operation."""
        if operation not in self.metrics:
            return {}
        
        metrics = self.metrics[operation]
        durations = [m.duration_ms for m in metrics]
        throughputs = [m.throughput for m in metrics]
        
        return {
            'count': len(metrics),
            'total_ms': sum(durations),
            'mean_ms': statistics.mean(durations),
            'median_ms': statistics.median(durations),
            'p95_ms': statistics.quantiles(durations, n=20)[18],
            'p99_ms': statistics.quantiles(durations, n=100)[98],
            'mean_throughput': statistics.mean(throughputs),
            'total_items': sum(m.item_count for m in metrics)
        }
    
    def print_report(self):
        """Print performance report."""
        print("\n=== Performance Report ===")
        for operation in sorted(self.metrics.keys()):
            stats = self.get_stats(operation)
            print(f"\n{operation}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean_ms']:.2f}ms")
            print(f"  Median: {stats['median_ms']:.2f}ms")
            print(f"  P95: {stats['p95_ms']:.2f}ms")
            print(f"  P99: {stats['p99_ms']:.2f}ms")
            print(f"  Throughput: {stats['mean_throughput']:.1f} items/sec")
```

## Benchmarking Strategy

### Performance Test Suite

```python
# test/test_performance.py
import pytest
import asyncio
import time
from pathlib import Path

@pytest.fixture
async def large_dataset(tmp_path):
    """Create large dataset for testing."""
    # Create 1000 test files
    for i in range(1000):
        model = create_test_model(i)
        file_path = tmp_path / f"model_{i}.fm.json"
        file_path.write_text(model.json())
    return tmp_path

@pytest.mark.benchmark
async def test_batch_processing_performance(large_dataset, benchmark):
    """Benchmark batch processing."""
    processor = BatchProcessor(max_concurrency=20)
    file_processor = OptimizedFileProcessor(processor)
    
    async def process():
        return await file_processor.process_directory_optimized(large_dataset)
    
    result = await benchmark(process)
    assert len(result) == 1000

@pytest.mark.benchmark
async def test_validation_performance(large_dataset, benchmark):
    """Benchmark validation performance."""
    models = load_test_models(large_dataset)
    service = ValidationService()
    
    async def validate():
        return await service.validate_batch(models)
    
    result = await benchmark(validate)
    assert len(result) == len(models)

@pytest.mark.benchmark 
async def test_search_performance(populated_index, benchmark):
    """Benchmark search performance."""
    search_engine = OptimizedSearchEngine(populated_index.repository, QueryCache())
    
    queries = ["pulmonary", "thyroid", "embolism", "nodule"] * 25  # 100 queries
    
    async def search():
        return await search_engine.search_batch_optimized(queries)
    
    result = await benchmark(search)
    assert len(result) == 100
```

## Implementation Timeline

### Phase 1: Infrastructure (Day 1-2)
- Implement connection pooling
- Add query caching
- Set up performance monitoring

### Phase 2: Database Optimization (Day 3-4)
- Implement batch operations
- Add query optimization
- Create proper indexes

### Phase 3: Concurrent Processing (Day 5-6)
- Implement BatchProcessor
- Add async file operations
- Optimize directory processing

### Phase 4: Search Optimization (Day 7)
- Add text indexes
- Implement search caching
- Optimize batch search

### Phase 5: Testing & Benchmarking (Day 8-9)
- Create performance test suite
- Run benchmarks
- Tune parameters
- Document results

## Success Metrics

### Performance Targets
- [ ] Batch validation: 40% faster
- [ ] File processing: 60% faster
- [ ] Search operations: 30% faster
- [ ] Database queries: 50% reduction

### Resource Usage
- [ ] Memory usage: < 500MB for 10k models
- [ ] CPU usage: Efficient multi-core utilization
- [ ] Network I/O: Minimized round trips
- [ ] Disk I/O: Optimized with async operations

### Quality Metrics
- [ ] All existing tests pass
- [ ] No functional regressions
- [ ] Performance tests in CI/CD
- [ ] Monitoring in production

## Risk Mitigation

### Performance Regression
- Comprehensive benchmarking before/after
- Performance tests in CI pipeline
- Feature flags for gradual rollout
- Easy rollback if issues detected

### Resource Exhaustion
- Configurable concurrency limits
- Rate limiting for external APIs
- Memory usage monitoring
- Circuit breakers for failures

### Data Consistency
- Maintain ACID properties
- Proper error handling in batch operations
- Transaction support where needed
- Validation of optimized paths

## Next Steps

1. **Benchmark Current**: Establish baseline metrics
2. **Implement Phase 1**: Core infrastructure
3. **Measure Impact**: Compare with baseline
4. **Iterate**: Tune based on results
5. **Document**: Update performance guidelines
6. **Monitor**: Set up production monitoring
7. **Optimize Further**: Based on real usage patterns