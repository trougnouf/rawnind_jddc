Pipeline Flow Diagram
═══════════════════════════════════════════════════════════════

┌─────────────────┐
│  DataIngestor   │----O |  Loads cached/remote indexes
└──────┬──────────┘
       ▼ <scene_send> (SceneInfo)
┌─────────────────┐
│   FileScanner   │ Looks for on-disk files
└────┬────────┬───┘  (resume download)       
     │        └───────────┐
     │ new_file_send      │ <missing_send>
     │ (ImageInfo)        ▼ (ImageInfo)
     │               ┌────────────┐
     │               │            │-concurrent----O    │
     │               │ Downloader │----downloads------O│
     │               │            │----------O         │
     │               └──────┬─────┘
     │          (ImageInfo) │ <downloaded_send>
     │          ┌───────────┘
     ▼          ▼
┌─────────────────────┐
│   Merge Channels    │
└───────────┬─────────┘
(ImageInfo) ▼ pre-existing + new downloads
┌─────────────────┐
│    Verifier     │                  
│     (sha1)      │
└────┬────────┬───┘
     │        │ <missing_send> (retry)
     │        └─────────────────┘
     │ <verified_send>          
     ▼ ( ImageInfo )                                      
┌─────────────────┐
│  SceneIndexer   │  Accumulates ImageInfos,
└────────┬────────┘
         ▼ <complete_scene_send> (SceneInfo)
┌───────────────────┐
│ MetadataArtificer │   Computes alignment, gain, masks
└────────┬──────────┘
         ▼ <enriched_send> (SceneInfo w/ metadata)
┌─────────────────┐
│AsyncCropProducer│-----------X crops.png
└─────────────────┘


Async Pipeline Classes:
───────────────
• DataIngestor:      Loads YAML index + JSON metadata from cache/remote
• FileScanner:       Checks filesystem for files (gt/ and scene dirs)
• Downloader:        Async HTTP downloads (max 5 concurrent)
• Verifier:          SHA-1 validation with retry loop for corrupted files
• SceneIndexer:      Tracks image arrival, emits complete scenes
• MetadataArtificer:      CPU-intensive computations (alignment, gain, masks)
• AsyncPipelineBridge
• 
•

Data Types:
───────────
• SceneInfo:  Collection of clean + noisy ImageInfo objects
• ImageInfo:  Individual image with filename, sha1, local_path, metadata


Pipeline Modes:
───────────────
1. With Enrichment:    Full pipeline including MetadataArtificer
2. Without Enrichment: Skips enrichment, goes straight to final consumer
