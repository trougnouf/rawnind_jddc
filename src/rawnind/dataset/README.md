# rawnind.dataset — Async Data Acquisition Pipeline

## Backgrounder

This module implements a concurrent pipeline for downloading, validating, and preprocessing the RawNIND dataset from remote storage. The architecture reflects lessons learned from earlier synchronous approaches: downloading gigabytes of image data sequentially is painfully slow, but naïve parallelization creates its own problems—overwhelming remote servers, exhausting file descriptors, running out of memory from buffering too many downloads simultaneously.

The solution is structured concurrency using the Trio async framework. The pipeline decomposes the acquisition process into discrete stages, each with bounded concurrency, communicating via typed channels. This design makes resource usage predictable, enables graceful degradation under failure, and provides clear points for progress monitoring.

### The Dataset Acquisition Problem

RawNIND consists of roughly 1000 image pairs (clean + noisy captures), totaling several hundred gigabytes. The images are stored on a remote Dataverse repository as individual files. Users need the dataset locally for training, but downloading everything on first run is impractical—it would take hours and consume significant bandwidth.

An ideal system would:
1. Download only what's actually needed (selective subset, or on-demand per-image)
2. Resume interrupted downloads without re-downloading completed files
3. Validate data integrity (detect corruption from transmission errors or disk failures)
4. Parallelize downloads to saturate network bandwidth
5. Compute metadata (alignment, masks) concurrently with downloads
6. Handle transient failures gracefully (retry with backoff)
7. Provide progress visibility for long-running operations

Traditional approaches struggle with these requirements. A sequential script is simple but slow. Shell scripting with GNU Parallel achieves parallelism but makes error handling and state management difficult. Custom multithreading code works but requires careful lock management and is hard to reason about.

Structured concurrency—where concurrency boundaries are explicit and tasks have clear parent-child relationships—addresses these issues. The Trio framework enforces that concurrent tasks are spawned within nurseries, ensuring they complete (or are cancelled) before the nursery exits. This prevents leaked background tasks and makes control flow comprehensible.

### Pipeline Architecture

The pipeline consists of stages connected by typed channels (Trio memory channels with bounded buffers). Each stage runs concurrently, reading from input channels and writing to output channels. Backpressure propagates automatically—if a downstream stage processes slowly, its input channel fills up, blocking the upstream sender until buffer space becomes available.

**Stage 1: DataIngestor**
Loads the dataset index (YAML file mapping image IDs to metadata) and scene descriptors (JSON files with detailed capture parameters). It checks the local cache first, falling back to HTTP download if missing. The ingestor produces SceneInfo objects—collections of ImageInfo describing all images in a scene (one clean exposure + multiple noisy exposures).

The index contains SHA-1 hashes for every file, enabling integrity verification. It also includes download URLs, expected file sizes, camera metadata (white balance, exposure time), and scene identifiers.

**Stage 2: FileScanner**
Receives SceneInfo objects and checks the local filesystem for each referenced image. Files already present (and verified on previous runs) are routed directly to downstream processing. Missing or unverified files are sent to the download queue.

This enables resume-after-interruption: if a previous run downloaded 500 files before terminating, the next run skips re-downloading those 500.

**Stage 3: Downloader**
Performs async HTTP downloads with bounded concurrency (default: 5 simultaneous downloads). Each worker pulls from the download queue, streams the file to disk, and emits an ImageInfo indicating the completed download.

Bounded concurrency prevents overwhelming the remote server (which may rate-limit or blacklist aggressive clients) and limits memory usage from buffering responses. The limit of 5 is empirically reasonable for most networks—high enough to saturate typical bandwidth, low enough to avoid congestion.

**Stage 4: Verifier**
Computes SHA-1 hashes of downloaded files and compares against expected values from the index. Verification failures indicate corruption (transmission errors, disk problems, or hash mismatches in the index itself).

Failed verification sends the file back to the download queue for retry (up to 3 attempts). Persistent failures after retries are logged as errors, but the pipeline continues processing other files.

**Stage 5: SceneIndexer**
Accumulates individual ImageInfo objects until all images for a scene have arrived. When a scene is complete (all clean + noisy captures present and verified), it emits a complete SceneInfo to the next stage.

This buffering stage transforms the per-image stream into a per-scene stream. Downstream processing often requires entire scenes (e.g., aligning multiple noisy captures to the clean reference), so this consolidation simplifies later stages.

**Stage 6: MetadataArtificer**
Computes CPU-intensive metadata for each scene: alignment shifts between clean/noisy pairs, gain normalization factors, validity masks, and pre-extracted crops at standard sizes. This preprocessing is optional—skipping it produces a lightweight pipeline that just downloads and validates files.

The enricher runs with bounded parallelism (default: 4 workers), reflecting CPU rather than I/O limitations. Alignment computation involves FFT operations or exhaustive spatial searches, both compute-bound. Running too many in parallel degrades performance (cache thrashing, CPU contention).

### Channel-Based Communication

Stages communicate through typed channels carrying SceneInfo or ImageInfo objects. Channels have bounded buffer sizes (typically 10-20 items), creating backpressure. If the verifier processes faster than the downloader, the channel between them fills, blocking the verifier until downloads catch up. This prevents memory exhaustion from unbounded queuing.

The channel abstraction simplifies error handling. If a stage crashes, Trio's nursery cancels all sibling tasks and propagates the exception upward. No complex cleanup logic is needed—incomplete downloads are abandoned, partial state is discarded, and the exception explains what failed.

### Concurrency Limits and Resource Management

Different stages have different concurrency limits reflecting their bottlenecks:

- **DataIngestor**: Single task (no parallelism benefit for reading an index file)
- **FileScanner**: Single task (filesystem metadata queries are fast, parallelism doesn't help)
- **Downloader**: 5 concurrent tasks (I/O bound, limited by network bandwidth and server politeness)
- **Verifier**: Single task (hashing is fast, parallelism complicates state tracking)
- **SceneIndexer**: Single task (bookkeeping, not a bottleneck)
- **MetadataArtificer**: 4 concurrent tasks (CPU bound, limited by core count)

These limits keep memory usage and file descriptor counts bounded. The worst-case simultaneous operations: 5 downloads (each holding an open HTTP connection and file handle) + 4 metadata computations (each loading multiple images into memory). This is tractable on commodity hardware.

### Error Handling and Retry Logic

Network operations fail. HTTP requests timeout, connections drop, servers return 5xx errors. The pipeline must handle transient failures without crashing.

The downloader implements retry-with-exponential-backoff: after a failed download, wait (initially 1 second, doubling on each retry) before retrying. After 3 attempts, mark the file as failed and continue. This tolerates transient network glitches without getting stuck in infinite retry loops.

Verification failures trigger re-downloads, on the assumption that corruption occurred during transmission. If re-downloading produces the same hash mismatch, either the file is corrupted at the source or the index has wrong metadata. Either way, the pipeline logs the failure and proceeds—one bad file shouldn't block processing hundreds of good files.

Permanent errors (malformed index file, inaccessible base directory) terminate the pipeline immediately, since they indicate configuration problems that won't resolve through retries.

### Progress Monitoring and Observability

Long-running downloads need progress visibility. The pipeline tracks:
- Total files to download vs. already present
- Active downloads and their progress (bytes downloaded / total size)
- Completed verifications
- Scenes processed through enrichment

This state is exposed through periodic log messages and (optionally) a real-time progress display. For batch processing, logging suffices. For interactive use, a terminal UI showing active transfers and completion percentages improves user experience.

The smoke test (`smoke_test.py`) demonstrates the progress monitoring, limiting downloads/processing to small subsets for rapid validation.

### Why Trio Instead of asyncio

Python's standard asyncio library could implement similar patterns, but Trio offers several advantages:

**Structured concurrency**: Trio enforces that tasks are spawned in nurseries and complete before the nursery exits. This prevents "fire-and-forget" tasks that run indefinitely, leak resources, or crash silently. In asyncio, background tasks can outlive their parent, leading to subtle bugs.

**Cancellation semantics**: Trio's cancellation is scope-based. Cancelling a nursery cancels all tasks within it atomically. asyncio's cancellation is per-task and requires careful exception handling to avoid partial completion states.

**Backpressure primitives**: Trio's memory channels have built-in buffering and blocking semantics. asyncio's queues exist but lack the clean blocking+cancellation interactions.

**Debuggability**: Trio's structured approach makes stack traces comprehensible. Debugging asyncio code often involves tracing through event loop internals.

For this application—concurrent I/O with clear stage boundaries—Trio's structured concurrency is a natural fit.

### Integration with Legacy Dataset Loaders

This async pipeline coexists with the older dataset loaders in `rawnind.libs.rawds`. The legacy loaders assume preprocessed data (OpenEXR files, YAML descriptors with alignment metadata) is already present. The async pipeline produces that data.

Typical workflow:
1. Run the async pipeline to download and verify files
2. Optionally run enrichment to compute alignment/masks
3. Use legacy dataset loaders during training to sample crops

Eventually, the dataset loaders may be refactored to use the async pipeline's outputs directly, eliminating the preprocessing step. Currently, the systems are decoupled for compatibility with existing training code.

### Memory and Disk Footprint

The pipeline is designed to stream data without holding the entire dataset in memory. At any moment, memory usage is bounded by:
- SceneInfo objects in channel buffers (~10-20 scenes × ~5 images × metadata size ≈ few MB)
- Active download buffers (5 concurrent × configurable chunk size ≈ few MB)
- MetadataArtificer working set (4 workers × 2-3 images × ~50 MB per image ≈ 400-600 MB)

Total memory usage stays under 1 GB even for large datasets, making the pipeline runnable on modest hardware.

Disk usage depends on the dataset size. RawNIND requires ~200 GB for raw files plus preprocessed OpenEXR. The pipeline doesn't create additional copies—files are downloaded to their final locations, and enrichment adds sidecar metadata files rather than duplicating images.

### Testing and Validation

The `smoke_test.py` script validates the pipeline with configurable limits (e.g., "download only the first 5 scenes" or "process only scenes from camera X"). This enables:
- Rapid iteration during development (don't wait hours to test changes)
- CI/CD integration (test pipeline correctness without downloading full dataset)
- Debugging specific failure modes (isolate problematic scenes)

The test configures the pipeline with short timeouts, limited concurrency, and verbose logging, making failures obvious.

### Performance Characteristics

On a typical broadband connection (100 Mbps), the pipeline achieves:
- Download throughput: ~80 Mbps sustained (below theoretical maximum due to server limits and protocol overhead)
- Verification: ~500 MB/s (SHA-1 hashing is fast on modern CPUs)
- Metadata enrichment: ~20-30 scenes/minute (varies with alignment complexity)

The limiting factor is usually network bandwidth for downloads and CPU for enrichment. Disk I/O rarely bottlenecks (modern SSDs sustain >500 MB/s sequential writes).

Total time to download and preprocess RawNIND from scratch: 4-8 hours depending on network and hardware.

### Future Directions

Potential enhancements:
- **Resumable HTTP**: Use byte-range requests to resume partial downloads (currently, interrupted downloads restart from zero)
- **Compression**: Download compressed archives instead of individual files (reduces transfer time, increases server load)
- **Differential sync**: Only download files changed since last sync (requires server-side modification time tracking)
- **On-demand loading**: Integrate with training loop to download/preprocess images just before they're needed (reduces initial wait, complicates training code)

The current design prioritizes simplicity and robustness over maximum performance. The pipeline is fast enough for research purposes—waiting a few hours for initial setup is acceptable when experiments take days to run.

## Module Organization

- **`DataIngestor.py`** — Loads dataset indexes from cache or remote
- **`FileScanner.py`** — Checks filesystem for existing files
- **`Downloader.py`** — Async HTTP downloads with concurrency limits
- **`Verifier.py`** — SHA-1 validation and retry logic
- **`SceneIndexer.py`** — Accumulates images into complete scenes
- **`MetadataArtificer.py`** — Computes alignment, gains, masks (optional stage)
- **`PipelineBuilder.py`** — Constructs and configures the full pipeline
- **`SceneInfo.py`** — Data structures for scenes and images
- **`smoke_test.py`** — Validation script with configurable limits
- **`pipeline_diagram.md`** — Visual flow diagram

## Related Documentation

- `src/rawnind/libs/rawds.py` — Legacy dataset loaders that consume this pipeline's output
- `src/rawnind/libs/rawproc.py` — Alignment and preprocessing algorithms used by MetadataArtificer