# Legacy File Analysis: src/rawnind/dataset/

**Date:** 2025-10-24
**Analysis Type:** Comprehensive codebase examination via parallel subagent dispatch
**Scope:** All Python files in `src/rawnind/dataset/` directory

---

## Legacy File Analysis: src/rawnind/dataset/

### Files Requiring Immediate Archival

#### **1. DiskCacheManager.py** - LEGACY ⚠️
**Status:** 914 lines of sophisticated caching infrastructure that was never integrated

**Evidence:**
- NOT exported from `__init__.py`
- NOT imported anywhere in the codebase (0 references found)
- NOT used in PipelineBuilder or orchestrator
- No test coverage

**Functionality:** Implements LRU/FIFO eviction policies, checksums, compression, connection pooling, self-healing cache, batch operations—production-ready features that suggest serious implementation effort but zero adoption.

**Recommendation:** Move to `archive/deprecated/_DEPRECATED_DiskCacheManager.py` with note: "Sophisticated disk-based caching implementation that was never integrated into the pipeline architecture. Superseded by simpler in-memory approaches."

---

#### **2. Aligner.py** - NAMING COLLISION ⚠️
**Status:** Contains old `MetadataArtificer` class that conflicts with newer implementation

**Critical Issue:** PipelineBuilder.py has naming collision:
```python
from .Aligner import MetadataArtificer              # Line 7 - artifact writer
from .MetadataArtificer import MetadataArtificer    # Line 11 - enricher (SHADOWS!)
```

Line 116 attempts to instantiate the shadowed class with wrong parameters, likely causing runtime errors.

**Evidence:**
- The `MetadataArtificer` in `Aligner.py` extends `PostDownloadWorker` (artifact writer)
- The `MetadataArtificer` in `MetadataArtificer.py` is standalone (enricher)
- Import at line 11 shadows line 7, breaking artifact writer usage

**Recommendation:**
1. Rename `Aligner.py::MetadataArtificer` → `AlignmentArtifactWriter` (matches git history commit dd5abed)
2. Or move entire `Aligner.py` to deprecated if artifact writing is handled elsewhere

---

### Files with Zero Usage (Candidates for Archival)

#### **3. AlignmentProgressTracker.py** - ORPHANED
- 0 imports found in active codebase
- Provides progress tracking infrastructure never adopted
- Move to `archive/unused/`

#### **4. cache_interfaces.py** - ORPHANED
- 0 imports found
- Defines `CacheInterface`, eviction policies (`LRUEvictionPolicy`, `FIFOEvictionPolicy`)
- Related to unused `DiskCacheManager.py`
- Move to `archive/unused/`

#### **5. channel_utils.py** - ORPHANED
- 0 imports found
- Provides `create_channel_dict()`, `merge_channels()` utilities
- Likely superseded by direct trio channel usage
- Move to `archive/unused/`

#### **6. collate_functions.py** - ORPHANED
- 0 imports found
- PyTorch DataLoader collate functions never integrated
- Contains `scene_batch_collate_fn`, `random_crop_collate_fn`, etc.
- Move to `archive/unused/`

#### **7. collate_strategies.py** - ORPHANED
- 0 imports found
- Factory pattern for collate strategies (`CollateStrategyFactory`)
- Never adopted in favor of simpler approaches
- Move to `archive/unused/`

#### **8. orchestrator.py** - UNDERUTILIZED
**Status:** Production-ready but not currently used

**Evidence:**
- 0 imports in active code (only referenced in git commit messages)
- 772 lines of state machine orchestration, metrics, retry logic
- Designed for production monitoring but smoke tests prefer manual wiring

**Nuance:** This is **intentionally unused** in development but valuable for production. Not deprecated, just not needed for current testing workflows.

**Recommendation:** Keep but document as "production-ready orchestration layer not required for development workflows."

---

### Files with Minimal/Internal-Only Usage

#### **9. MeilisearchIndexer.py** - OPTIONAL
- Only imported in its own test file
- NOT used in PipelineBuilder (despite being designed as optional postprocessor)
- Provides search indexing for dev/QA but never integrated

**Status:** Optional component that could be useful but isn't currently wired up.

**Recommendation:** Either integrate into PipelineBuilder as optional postprocessor or move to `archive/optional_components/` with usage documentation.

---

#### **10. adapters.py** - LEGACY COMPATIBILITY
**Status:** Contains `LegacyAdapter` and `BackwardsCompatAdapter` classes

**Evidence:**
- Exported from `__init__.py` but never actually imported elsewhere
- Provides migration path from old YAML-based datasets
- 1 commit in last 6 months (minimal maintenance)

**Recommendation:** Keep during transition period but add deprecation warnings. Document migration timeline.

---

#### **11. visualizer.py** - TEST-ONLY
**Status:** Only used in smoke tests, not production pipeline

**Evidence:**
- Used in `tests/smoke_test.py` and `tests/smoke_test_async_crops.py`
- 171 lines with recent enhancements (counter improvements)
- Valuable for development but not production infrastructure

**Recommendation:** Keep in current location—it's actively maintained test infrastructure, not legacy.

---

### Active Pipeline Components (Keep)

The following are **ACTIVE** and core to current architecture:

| File | Role | Status |
|------|------|--------|
| **AsyncPipelineBridge.py** | Terminal consumer, trio→PyTorch bridge | Core active ✓ |
| **SceneInfo.py** | Data structures (SceneInfo, ImageInfo) | Core active ✓ |
| **PipelineBuilder.py** | Pipeline assembly and configuration | Core active ✓ |
| **MetadataArtificer.py** | Metadata enrichment stage | Core active ✓ |
| **CropProducerStage.py** | Async crop generation | Core active ✓ |
| **DataIngestor.py** | Stage 1: Dataset loading | Core active ✓ |
| **FileScanner.py** | Stage 2: File presence scanning | Core active ✓ |
| **Downloader.py** | Stage 3: HTTP downloads | Core active ✓ |
| **Verifier.py** | Stage 4: Hash validation | Core active ✓ |
| **SceneIndexer.py** | Stage 5: Scene assembly | Core active ✓ |
| **PostDownloadWorker.py** | Base class for postprocessors | Core active ✓ |
| **pipeline_decorators.py** | Stage decorator framework | Core active ✓ |
| **cache.py** | StreamingJSONCache implementation | Core active ✓ |
| **constants.py** | Configuration dataclasses | Core active ✓ |

---

## Summary and Recommendations

### Immediate Actions (High Confidence)

```bash
# Archive completely unused infrastructure
mv src/rawnind/dataset/DiskCacheManager.py archive/deprecated/_DEPRECATED_DiskCacheManager.py
mv src/rawnind/dataset/AlignmentProgressTracker.py archive/unused/
mv src/rawnind/dataset/cache_interfaces.py archive/unused/
mv src/rawnind/dataset/channel_utils.py archive/unused/
mv src/rawnind/dataset/collate_functions.py archive/unused/
mv src/rawnind/dataset/collate_strategies.py archive/unused/

# Fix naming collision in Aligner.py
# (Requires code changes - rename MetadataArtificer → AlignmentArtifactWriter)
```

### Document and Monitor

- **orchestrator.py**: Production-ready but not used in dev—document this explicitly
- **adapters.py**: Add deprecation warnings, plan removal timeline
- **MeilisearchIndexer.py**: Either integrate or document as optional component

### Total Files to Archive

- **Immediate archival:** 6 files (DiskCacheManager + 5 unused utilities)
- **Naming collision fix:** 1 file (Aligner.py needs refactoring)
- **Monitor for deprecation:** 2 files (adapters.py, MeilisearchIndexer.py)
- **Keep active:** 14 files (core pipeline components)

The analysis reveals that recent refactoring (commit dd850ec) successfully modernized the async pipeline but left several orphaned infrastructure experiments and one critical naming collision that should be addressed.

---

## Methodology

This analysis was conducted by dispatching five parallel subagents to examine:
1. Pipeline stage files (DataIngestor, FileScanner, Downloader, DiskCacheManager, SceneIndexer)
2. Metadata/processing files (MetadataArtificer, CropProducerStage, PostDownloadWorker, SceneInfo, AlignmentProgressTracker)
3. Infrastructure files (PipelineBuilder, orchestrator, AsyncPipelineBridge, MeilisearchIndexer, pipeline_decorators)
4. Archive patterns and deprecation markers
5. Cross-reference import analysis across entire codebase

Each subagent used Read, Grep, and code analysis tools to determine:
- Import counts and locations
- Usage patterns in tests vs production
- Deprecation markers and TODO comments
- Git history and recent modifications
- Naming collisions and architectural conflicts