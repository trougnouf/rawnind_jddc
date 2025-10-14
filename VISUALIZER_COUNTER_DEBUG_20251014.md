# Visualizer Counter Investigation - 2025-10-14

## Problem

The smoke_test visualizer bottom section shows stuck counters:
```
│ Bridge       │ Complete:   0
└──▼
│ DataLoader   │ Scenes:   0 │ Batches:   0 │ ✗
```

These remain at zero even though the crop producer shows progress ("Cropped: 8").

## Architecture

### Pipeline Flow

```
CropProducer → cropped_send → cropped_recv → AsyncPipelineBridge → _scenes list
                                                       ↓
                                           _watch_bridge() polls len(bridge)
                                                       ↓
                                           viz.update(complete=Δ)
```

### Key Components

**AsyncPipelineBridge (AsyncPipelineBridge.py:395-420)**
- Consumes scenes from `cropped_recv` channel
- Validates and stores scenes in `self._scenes` list
- `len(bridge)` returns `len(self._scenes)`

**_watch_bridge() (smoke_test.py:491-503)**
- Polls `len(bridge)` every 0.1 seconds
- Calculates delta: `n - last`
- Updates visualizer: `viz.update(complete=(n - last))`

**test_dataloader_integration() (smoke_test.py:106-206)**
- Triggered when `n >= 2` scenes collected (line 508)
- Writes YAML cache file
- Creates PyTorch Dataset and DataLoader
- Updates various `dataloader_*` counters

## Diagnostic Logging Added

### 1. Bridge Consumption Tracking

**File**: `src/rawnind/dataset/AsyncPipelineBridge.py:397-411`

Added logging to show:
- When bridge receives items from channel
- When scenes pass validation
- When scenes are successfully collected
- Current count in `_scenes` list

```python
logger.debug(f"Bridge received item: {item.scene_name}")
logger.info(f"Bridge collected scene {collected}: {item.scene_name} (total: {len(self._scenes)})")
logger.warning(f"Scene processing returned False for {item.scene_name}")
```

### 2. Watch Function Diagnostics

**File**: `tests/smoke_test.py:497-503`

Added logging to show:
- When bridge count increases (scene collection events)
- When bridge is empty despite cropping activity (potential issue)
- Bridge state for debugging

```python
logging.info(f"Bridge progress: {n} scenes collected (was {last})")
logging.warning(f"Bridge empty despite {viz.counters['cropped']} scenes cropped. Bridge state: {bridge.state.value}")
```

## Possible Root Causes

### 1. Scene Validation Failure

Bridge validates scenes before storing (AsyncPipelineBridge.py:447-448):
```python
if self.validate_scenes and not self._validate_scene(item):
    return False
```

Validation checks:
- Has `scene_name` attribute
- Has `clean_images` and `noisy_images` attributes

If validation fails, scenes are silently dropped. **Now logged as warning.**

### 2. Scene Filtering

Bridge can filter scenes (AsyncPipelineBridge.py:451-454):
```python
if self._should_filter_scene(item):
    if self.stats:
        self.stats.scenes_filtered += 1
    return False
```

Filters:
- Test reserve scenes (if `filter_test_reserve=True`)
- CFA type mismatch (if `cfa_filter` is set)

Current smoke_test config sets neither, so filtering should not occur.

### 3. Channel Connection Issue

If `cropped_recv` channel not properly connected:
- Bridge receives nothing
- `len(bridge)` stays at 0
- `_watch_bridge` sees no progress

The channel is created as unbuffered (buffer_size=0):
```python
cropped_send, cropped_recv = trio.open_memory_channel(0)
```

This means sends block until received - should prevent data loss.

### 4. Timing Race Condition

Possible sequence:
1. Cropper finishes, closes `cropped_send`
2. Bridge consumes all scenes rapidly
3. Bridge.consume() exits
4. _watch_bridge samples `len(bridge)` too late
5. Sees final count but visualizer wasn't updated

**Fixed by logging**: Now logs every collection event.

### 5. Exception Swallowing

If bridge encounters exception during processing:
```python
except Exception as e:
    if self.retry_on_error and retry_count < self.DEFAULT_RETRY_COUNT:
        retry_count += 1
        logger.warning(f"Error processing scene (retry {retry_count}): {e}")
        await trio.sleep(self.DEFAULT_RETRY_DELAY_SECONDS)
        continue
    else:
        logger.error(f"Failed to process scene after {retry_count} retries: {e}")
```

After 3 retries, scene is dropped but error is logged.

## How to Debug

### 1. Check Bridge Logs

Look for these patterns in `/tmp/smoke_test.log`:

**Expected (healthy):**
```
Bridge received item: scene_name_001
Bridge collected scene 1: scene_name_001 (total: 1)
Bridge progress: 1 scenes collected (was 0)
```

**Problem: Validation failure:**
```
Bridge received item: scene_name_001
Scene processing returned False for scene_name_001
```

**Problem: Not receiving:**
```
(no "Bridge received" messages despite "Cropped: N" > 0)
```

### 2. Check Bridge State

The warning at smoke_test.py:503 will fire if:
- Cropping has happened (`cropped > 0`)
- But bridge is empty (`len(bridge) == 0`)

Shows bridge state to distinguish:
- `INITIALIZED`: Bridge created but consume() not called yet
- `COLLECTING`: Currently consuming
- `READY`: Consumption finished
- `ERROR`: Exception occurred

### 3. Verify Scene Structure

If validation is failing, check SceneInfo objects:
```python
# In cropper_with_memory_tracking (smoke_test.py:323)
logging.debug(f"Scene structure: name={scene.scene_name}, "
              f"clean_images={len(scene.clean_images)}, "
              f"noisy_images={len(scene.noisy_images)}")
```

### 4. Monitor Channel Flow

Add logging to cropper send:
```python
# smoke_test.py:331
logging.info(f"Sending scene {scene.scene_name} to bridge channel")
await channels['cropped_send'].send(processed)
```

Should see 1:1 correspondence with bridge receives.

## DataLoader Counter Fix

The DataLoader counters depend on:
1. Bridge collecting >= 2 scenes
2. YAML cache being written
3. test_dataloader_integration executing

If bridge counters fix doesn't fix dataloader counters:
- Check if `n >= 2` condition at line 508 is reached
- Check if YAML write succeeds (line 512)
- Check if test runs (look for "Testing DataLoader + Model Integration" output)
- Check for dataloader exceptions in logs

## Next Steps

1. **Run smoke test with new logging**
   ```bash
   python tests/smoke_test.py --max-scenes 3 --timeout 300
   ```

2. **Examine `/tmp/smoke_test.log`** for bridge patterns above

3. **If bridge not receiving**: Channel connection issue - check cropper → bridge wiring

4. **If receiving but not collecting**: Validation or filtering issue - check scene structure

5. **If collecting but viz not updating**: _watch_bridge timing issue - but logging will show collection events

## Files Modified

- `src/rawnind/dataset/AsyncPipelineBridge.py` (lines 397, 407, 411)
- `tests/smoke_test.py` (lines 498, 501-503)

## Errata: Additional Changes Not Documented Above

While adding the diagnostic logging, several ancillary fixes were made to smoke_test.py:

### CLI Argument Fixes

**Missing `--max-scenes` parameter** (lines 568-570):
- Previously absent from argument parser, causing failures when used
- Now properly defined with help text: "Maximum scenes allowed 'in flight' through the pipeline during processing."
- This parameter was referenced in the example command above but wasn't actually parseable

**Timeout type correction** (line 572):
- Changed from `type=float` to `type=int`
- Aligns with actual usage (timeout in whole seconds)

**Docstring clarification** (line 362):
- Updated `max_scenes` description to clarify it controls "in flight" limit during processing
- Not a total count of scenes to process, but a concurrency/memory management parameter

### Architectural TODO Comments

**Line 406**: "Separation of concerns: the following needs to be encapsulated and moved into PipelineBuilder"
- Notes that manual stage wiring should migrate to builder pattern

**Lines 505-508**: Trigger mechanism critique
- Current push-based approach (watch bridge, trigger dataloader test) flagged as incorrect
- Proposes pull-based BatchAccumulator pattern where training loop pulls batches on demand
- "The training loop needs to be able to trigger for a single batch but I think we are basically there."

These changes support the diagnostic infrastructure but weren't part of the core counter debugging investigation.


## Async Crop Producer Variant

Created `tests/smoke_test_async_crops.py` as a variant using `CropProducerStageAsync` instead of `CropProducerStage`.

**Key differences:**
- Uses `consume_and_produce()` pattern instead of manual wrapper with `process_scene()`
- Trio-native with operation-level parallelism (MS-SSIM per-window, concurrent crops)
- Built-in memory management (no manual unload wrapper needed)
- No ProcessPoolExecutor serialization overhead
- Logs to `/tmp/smoke_test_async_crops*.log` for parallel testing

Expected benefits: lower memory footprint, better CPU utilization through Trio's scheduler.

Purpose: Compare memory usage and identify whether crop production issues are architecture-specific or data-flow inherent.
