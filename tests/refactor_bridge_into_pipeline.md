### The Plan
'Below is a map of the moving parts, their responsibilities and boundaries, the exact OOM vector, and the concrete 
steps 
to remove `YAMLArtifactWriter` and replace it with an in-memory `AsyncPipelineBridge` (legacy interface only).

---

### What lives in `src/rawnind/dataset/` and what each piece does

- `__init__.py`
  - Re-exports top-level components for easy imports: `DataIngestor`, `FileScanner`, `Downloader`, `Verifier`, `SceneIndexer`, `AsyncAligner`, `PipelineBuilder`, `SceneInfo`/`ImageInfo`.

- `SceneInfo.py`
  - Defines `ImageInfo` and `SceneInfo` dataclasses and the tensor cache contract.
  - Key methods:
    - `ImageInfo.load_image(as_torch=True)`: loads RAW via `rawnind.libs.rawproc.img_fpath_to_np_mono_flt_and_metadata`, caches tensor (torch or numpy) and merges returned metadata.
    - `ImageInfo.unload_image()`: releases cached tensor to limit memory pressure.
    - `ImageInfo.aligned_image_tensor`: view producing a cropped/shifted aligned tensor (no allocation beyond slicing semantics).

- `channel_utils.py`
  - Utilities for wiring Trio channels: `create_channel_dict`, `merge_channels`, `limit_producer`, `consume_until`.
  - Important: It’s used in `tests/smoke_test.py` to limit and terminate the pipeline.

- `visualizer.py`
  - Live terminal HUD with counters for each stage, including keys for `yaml_writing/yaml_written` which we’ll retire once YAML is removed.

- `DataIngestor.py`
  - Source of truth for scene metadata ingestion. Builds `SceneInfo` and `ImageInfo` from index files (local or fetched remote YAML/JSON).
  - Produces scenes to a channel.

- `FileScanner.py`
  - Given a stream of `SceneInfo`, checks filesystem for expected files under `dataset_root/<cfa>/<scene_name>`.
  - Sends existing files to `new_file_send`; missing ones to `missing_send` (for downloader).

- `Downloader.py`
  - Concurrent async HTTP downloader using `httpx`, with retry logic and optional progress bar.
  - Opportunistically calls `img_info.load_image(as_torch=True)` after download, pre-populating the image cache for locality.

- `Verifier.py`
  - Recomputes SHA-1 on disk, compares with expected; routes valid files to `verified_send` or back to `missing_send` on failure (with retry cap).

- `SceneIndexer.py`
  - Aggregates `ImageInfo` into complete `SceneInfo` once all SHA1s for a scene are present; then emits complete scenes.

- `AsyncAligner.py`
  - Adds derived metadata to `ImageInfo.metadata` (alignment estimates, gains, masks, color matrices, etc.). Also supports an optional “crops list” enrichment path when `enable_crops_enrichment` is True.
  - In smoke test it’s instantiated with `enable_crops_enrichment=False`, i.e., we expect downstream stages (MetadataArtificer/Cropper) to produce real crop metadata.

- `pipeline_decorators.py`
  - Provides `@stage` decorator used by the post-download stages. It increments visualizer counters (e.g., `('cropping','cropped')`). This is how `MetadataArtificer` and `CropProducerStage` report progress.

- `PostDownloadWorker.py`
  - Abstract base for CPU/post-download stages. Notable features:
    - Async context manager (`__aenter__/__aexit__`) to spin up and tear down a `ProcessPoolExecutor` if `use_process_pool=True`.
    - `consume_and_produce(recv, send)` orchestrates the read/transform/write loop.
    - `run_cpu_bound(func, ...)` executes a static CPU-heavy function in the process pool.

- `MetadataArtificer.py`
  - `PostDownloadWorker` that writes optional alignment artifacts (masks, per-pair YAML) and populates alignment-related metadata on `ImageInfo.metadata`.
  - Uses `@stage(progress=("aligning","aligned"))` for progress.

- `crop_producer_stage.py` (`CropProducerStage`)
  - `PostDownloadWorker` that converts cached tensors to numpy, unloads them in the main process (critical for memory), then offloads heavy crop extraction and saving into a process pool via static `_extract_and_save_crops`.
  - After the pool returns crop descriptors, it attaches them into `noisy_img.metadata['crops']`.
  - This stage is memory-conscious: it explicitly calls `gt_img.unload_image()` and `noisy_img.unload_image()` right after conversion to numpy and before dispatch to the process pool.

- `YAMLArtifactWriter.py`
  - Current terminal consumer in `tests/smoke_test.py` chain.
  - Problem: Buffers all scene descriptors in `self.descriptors` and writes once in `shutdown()`. This can OOM on longer runs because the list grows unbounded and YAML serialization materializes a giant structure in memory.

- `PipelineBuilder.py`
  - A higher-level orchestrator that can assemble the pipeline with or without enrichment and with pluggable postprocessors like `MetadataArtificer` and `CropProducerStage`.

- `orchestrator.py`
  - A more fully-featured orchestrator (stateful) intended to coordinate the pipeline; currently not used by the smoke test.

---

### How the smoke test wires these pieces today

File: `tests/smoke_test.py`
- Constructs the stages and channels explicitly.
- Uses buffered channels for file-heavy parts (`scene/new_file/missing/...`) and unbuffered channels (`enriched/aligned/cropped/yaml`) to keep memory bounded at heavy stages.
- Terminal consumer is `YAMLArtifactWriter.consume_and_produce(...)`, followed by `consume_until(yaml_recv, max_scenes, ...)` to terminate.
- After pipeline completion, it tries to instantiate a real dataset that consumes the written YAML and loads a batch (sanity test).

This flow is compatible with an artifact-based workflow but, as observed, risks OOM due to YAML buffering in memory.

---

### The OOM vector (from the actual code)

- `YAMLArtifactWriter`:
  - Accumulates every scene’s descriptor in `self.descriptors` (list) inside `process_scene`.
  - Writes all descriptors to disk on `shutdown()`.
  - With unbounded scenes and crop metadata per scene, this list grows until the process runs out of RAM.

Even though `CropProducerStage` is careful to unload tensors and to do its heavy lifting in worker processes, the YAML writer’s all-in-RAM list becomes the limiting factor.

---

### Your requested change: remove YAML entirely and wire `CropProducerStage` → `AsyncPipelineBridge` (legacy only)

Here’s the correct, code-aware plan that touches all the right places.

#### 1) Move the bridge out of `archive/` and into dataset package

- Source file: `archive/newark_training_integration/async_to_sync_bridge.py`
- Target file: `src/rawnind/dataset/AsyncPipelineBridge.py`
- Edit imports inside the bridge:
  - Replace `from src.rawnind.dataset.SceneInfo import SceneInfo` with `from .SceneInfo import SceneInfo`.
- Optionally add an export to `src/rawnind/dataset/__init__.py`:
  - `from .AsyncPipelineBridge import AsyncPipelineBridge`

What the bridge must expose for our use now:
- Async consumer on the pipeline side, e.g., `async def consume(self, recv: trio.MemoryReceiveChannel)` that appends scenes into an internal list and respects `max_scenes`.
- Legacy synchronous interface on the data side (for now):
  - `__len__`, `__getitem__` that either return `SceneInfo` or a legacy RawNIND dict (toggle via `backwards_compat_mode=True`).

The file already includes a robust `AsyncPipelineBridge` class with thread-safety and stats. The only required change is import relocation plus adding or confirming a simple `consume(recv)` method that reads `SceneInfo` instances and appends them.

#### 2) Remove `YAMLArtifactWriter` from the pipeline wiring

In `tests/smoke_test.py`:
- Remove creation and context manager for `YAMLArtifactWriter`.
- Instantiate `AsyncPipelineBridge` instead.
- Wire it as the terminal consumer of `cropped_recv`.
- Update the end condition: have `consume_until(...)` read from a simple channel is no longer necessary if the bridge manages the count; instead, we can:
  - run a small watcher that checks `len(bridge)` and cancels the nursery when it reaches `max_scenes`, or
  - simply let `bridge.consume(recv)` break when `max_scenes` is reached and close.

Sketch of the minimal changes:

```python
# from rawnind.dataset.YAMLArtifactWriter import YAMLArtifactWriter
from rawnind.dataset.AsyncPipelineBridge import AsyncPipelineBridge

# ... inside run_smoke_test
bridge = AsyncPipelineBridge(
    max_scenes=max_scenes,
    backwards_compat_mode=True,  # expose legacy interface
)

async with aligner, cropper:  # <- YAMLArtifactWriter removed
    async with trio.open_nursery() as nursery:
        # ... start earlier stages as-is

        # Pipeline tail
        nursery.start_soon(
            aligner.consume_and_produce,
            channels['enriched_recv'], channels['aligned_send']
        )
        nursery.start_soon(
            cropper.consume_and_produce,
            channels['aligned_recv'], channels['cropped_send']
        )

        # Bridge as terminal consumer (in-memory, legacy sync interface)
        async def _bridge_consume():
            await bridge.consume(channels['cropped_recv'])
        nursery.start_soon(_bridge_consume)

        # Optional visual progress watcher to replace YAML counters
        async def _watch_bridge():
            last = 0
            while True:
                await trio.sleep(0.1)
                n = len(bridge)
                if n > last:
                    await viz.update(complete=(n - last))
                    last = n
                if max_scenes and n >= max_scenes:
                    break
        nursery.start_soon(_watch_bridge)
```

Notes:
- Keep `enriched/aligned/cropped` channels unbuffered (as they are) to maintain backpressure.
- `YAML` counters in the visualizer will remain at zero; that’s acceptable, or we can patch `visualizer.py` to add a new counter label for the bridge.

#### 3) Expose legacy shape via the bridge

Because `YAMLArtifactWriter`’s `scene_to_yaml_descriptor` defined the shape expected by the legacy datasets (e.g., `CleanProfiledRGBNoisyBayerImageCropsDataset`), the bridge needs to map `SceneInfo` to the same kind of dict.

You can literally reuse the function body from `YAMLArtifactWriter.scene_to_yaml_descriptor(scene, dataset_root)` and call it per `__getitem__`. The only change is we won’t write to disk anymore.

Minimal mapping inside the bridge:

```python
def _map_to_legacy(self, scene: SceneInfo) -> dict:
    gt_img = scene.get_gt_image()
    noisy_img = scene.noisy_images[0]
    md = noisy_img.metadata or {}
    return {
        "scene_name": scene.scene_name,
        "image_set": scene.scene_name,
        "is_bayer": scene.cfa_type == "bayer",
        "f_fpath": str(noisy_img.local_path or ""),
        "f_bayer_fpath": str(noisy_img.local_path or ""),
        "gt_fpath": str(gt_img.local_path or ""),
        "gt_linrec2020_fpath": str(gt_img.local_path or ""),
        "gt_bayer_fpath": str(gt_img.local_path or ""),
        "f_linrec2020_fpath": str(noisy_img.local_path or ""),
        "best_alignment": md.get("alignment", [0, 0]),
        "best_alignment_loss": md.get("alignment_loss", 0.0),
        "raw_gain": md.get("raw_gain", 1.0),
        "rgb_gain": md.get("rgb_gain", 1.0),
        "mask_mean": md.get("mask_mean", 1.0),
        "mask_fpath": md.get("mask_fpath", ""),
        "rgb_xyz_matrix": md.get("rgb_xyz_matrix", [[1,0,0],[0,1,0],[0,0,1]]),
        "overexposure_lb": md.get("overexposure_lb", 1.0),
        "crops": md.get("crops", []),
        "rgb_msssim_score": md.get("msssim_score", 1.0),
    }

def __getitem__(self, idx: int):
    scene = self._scenes[idx]
    return self._map_to_legacy(scene) if self.backwards_compat_mode else scene
```

That way, the training side that expected to read legacy YAML manifests can instead read from the bridge list in-memory and receive the same dicts.

#### 4) Keep memory bounded and concurrency sane

- `CropProducerStage` already unloads tensors before dispatching to the pool, which is correct.
- Removing YAML eliminates the `self.descriptors` RAM growth.
- If you still see pressure, temporarily reduce `DEFAULT_MAX_WORKERS` passed to `MetadataArtificer`/`CropProducerStage` in the smoke test from `~0.75 * CPU` to `1–2` to validate footprint.

#### 5) Optional: adjust the smoke test’s “DataLoader integration” section

The current test reads `tmp/rawnind_dataset/pipeline_output.yaml`. With YAML removed, this check will always skip. If you want to keep a similar end-to-end sanity check without writing to disk:

- Implement a tiny adapter dataset that takes the bridge and yields samples mapped to the legacy dict (or directly to tensors the model expects). Or,
- Write a one-off `NDJSON` dump at the end using the bridge’s in-memory list for debugging only (but do not accumulate during pipeline — only serialize after collection, bounded by `max_scenes`).

---

### Why this plan matches your code and your requirements

- It removes the only demonstrated unbounded RAM growth vector (`YAMLArtifactWriter.descriptors`).
- It respects your pipeline architecture: the heavy stages remain unbuffered and memory-conscious, and we keep `ProcessPoolExecutor` management via `PostDownloadWorker` context.
- It uses your existing archive bridge (moved in place), limiting scope.
- It keeps the legacy interface to avoid rippling changes into training code right now.

---

### Checklist to implement

- Move `archive/newark_training_integration/async_to_sync_bridge.py` → `src/rawnind/dataset/AsyncPipelineBridge.py` and fix the import of `SceneInfo`.
- Export `AsyncPipelineBridge` in `src/rawnind/dataset/__init__.py` (optional but convenient).
- Add a simple `consume(self, recv)` coroutine to the bridge if it doesn’t exist yet; append scenes; stop on `max_scenes`.
- Port over `scene_to_yaml_descriptor` logic into the bridge’s legacy mapper for `__getitem__`.
- Update `tests/smoke_test.py`:
  - Remove `YAMLArtifactWriter` and the `yaml_send/yaml_recv` channel.
  - Start the bridge consumer on `cropped_recv`.
  - Replace YAML progress with a length-based watcher that emits `viz.update(complete=Δ)`.
- Optionally trim `visualizer.py` to hide YAML counters.

If you want, I can draft specific diffs for `smoke_test.py`, the bridge file move, and a minimal mapping inside the bridge file next. Let me know if you want multi-document YAML streaming kept as a debug-only fallback — but per your instruction, I’ll proceed with a full removal of `YAMLArtifactWriter` from the pipeline path and wire the bridge as the terminal consumer with legacy interface only.