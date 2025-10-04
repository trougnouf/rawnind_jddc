Pipeline Flow Diagram
═════════════════════════════════════════════════════════════════════

┌─────────────────┐
│  DataIngestor   │  Loads cached/remote indexes
│                 │  Produces SceneInfo objects with ImageInfo
└────────┬────────┘
         │ scene_send (SceneInfo)
         ▼
┌─────────────────┐
│   FileScanner   │  Scans filesystem for existing files
│                 │  Routes found/missing ImageInfo
└────┬────────┬───┘
     │        │
     │        └─────────────────────────────────────────┐
     │ new_file_send                    missing_send    │
     │ (ImageInfo w/ local_path)        (ImageInfo)    │
     │                                                  │
     │                                                  ▼
     │                                         ┌─────────────────┐
     │                                         │   Downloader    │
     │                                         │  (async, 5x)    │
     │                                         └────────┬────────┘
     │                                                  │
     │                                                  │ downloaded_send
     │                                                  │ (ImageInfo)
     │                                                  │
     │          ┌───────────────────────────────────────┘
     │          │
     ▼          ▼
┌─────────────────────┐
│   Merge Channels    │  Combines new_file + downloaded
└──────────┬──────────┘
           │ merged (ImageInfo)
           ▼
┌─────────────────┐
│    Verifier     │  SHA-1 hash validation
│                 │  Retries up to 3x on failure
└────┬────────┬───┘
     │        │
     │        └──────────────────────────────────────────┐
     │ verified_send                    missing_send     │
     │ (ImageInfo, validated=True)      (corrupted)     │
     │                                                   │
     │                                          (retry loop)
     ▼
┌─────────────────┐
│  SceneIndexer   │  Accumulates images per scene
│                 │  Emits complete SceneInfo when all images arrive
└────────┬────────┘
         │ complete_scene_send (SceneInfo)
         ▼
┌─────────────────┐
│MetadataEnricher │  [OPTIONAL] Computes alignment, gain, masks
│    (async, 4x)  │  Enriches with crops metadata
└────────┬────────┘
         │ enriched_send (SceneInfo w/ metadata)
         ▼
┌─────────────────┐
│ Final Consumer  │  Logs completion / custom logic
└─────────────────┘


Key Components:
───────────────
• DataIngestor:      Loads YAML index + JSON metadata from cache/remote
• FileScanner:       Checks filesystem for files (gt/ and scene dirs)
• Downloader:        Async HTTP downloads (max 5 concurrent)
• Verifier:          SHA-1 validation with retry loop for corrupted files
• SceneIndexer:      Tracks image arrival, emits complete scenes
• MetadataEnricher:  CPU-intensive computations (alignment, gain, masks)


Data Types:
───────────
• SceneInfo:  Collection of clean + noisy ImageInfo objects
• ImageInfo:  Individual image with filename, sha1, local_path, metadata


Pipeline Modes:
───────────────
1. With Enrichment:    Full pipeline including MetadataEnricher
2. Without Enrichment: Skips enrichment, goes straight to final consumer
