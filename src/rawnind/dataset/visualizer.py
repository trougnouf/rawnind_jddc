"""Live terminal visualization for pipeline progress."""

import sys
import trio
from typing import Dict, Optional


class PipelineVisualizer:
    """Live visualization of pipeline stage counters with color coding."""

    def __init__(self, total_items: Optional[int] = None):
        self._use_colors = sys.stdout.isatty()
        self._init_colors()
        
        self.counters: Dict[str, int] = {
            'scanned': 0, 'found': 0, 'missing': 0, 'queued': 0,
            'active': 0, 'finished': 0, 'verifying': 0, 'verified': 0,
            'failed': 0, 'errors': 0, 'Indexed': 0, 'complete': 0,
            'enriching': 0, 'enriched': 0, 'aligning': 0, 'aligned': 0,
            'cropping': 0, 'cropped': 0, 'yaml_writing': 0, 'yaml_written': 0,
        }
        self.zero_since: Dict[str, float] = {}
        self.lock = trio.Lock()
        self._last_lines = 0
        self._start_time: Optional[float] = None
        self.total_items = total_items
        self.download_start_time: Optional[float] = None
        self.total_downloads = 0

    def _init_colors(self) -> None:
        """Initialize ANSI color codes based on terminal capability."""
        if self._use_colors:
            self.RESET = '\033[0m'
            self.GREEN = '\033[92m'
            self.YELLOW = '\033[93m'
            self.RED = '\033[91m'
            self.BLUE = '\033[94m'
            self.BOLD = '\033[1m'
        else:
            self.RESET = self.GREEN = self.YELLOW = ''
            self.RED = self.BLUE = self.BOLD = ''

    def _get_color(self, counter_name: str, has_downstream: bool = True) -> str:
        """Get color for a counter based on its value and stall time."""
        value = self.counters[counter_name]

        if value > 0:
            if counter_name in self.zero_since:
                del self.zero_since[counter_name]
            return self.GREEN

        current_time = trio.current_time()
        if counter_name not in self.zero_since:
            self.zero_since[counter_name] = current_time

        time_at_zero = current_time - self.zero_since[counter_name]
        return self.BLUE if time_at_zero > 1.5 else self.RESET

    def _format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS or MM:SS."""
        if seconds < 0:
            return "--:--"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours > 0 else f"{minutes:02d}:{secs:02d}"

    def _estimate_time_remaining(self) -> Optional[float]:
        """Estimate time remaining based on progress."""
        if self._start_time is None or self.total_items is None or self.total_items == 0:
            return None

        completed = self.counters['complete']
        if completed == 0:
            return None

        elapsed = trio.current_time() - self._start_time
        rate = completed / elapsed
        remaining_items = self.total_items - completed

        return remaining_items / rate if rate > 0 else None

    def _render(self) -> str:
        """Render the pipeline diagram with current counters and color coding."""
        if self._start_time is None:
            self._start_time = trio.current_time()

        elapsed = trio.current_time() - self._start_time
        elapsed_str = self._format_time(elapsed)

        pct = (self.counters['complete'] / self.total_items * 100) if self.total_items else 0
        eta_str = self._format_time(eta) if (eta := self._estimate_time_remaining()) else '--:--'
        
        header_content = f"{elapsed_str} │ {pct:4.1f}% │ ETA {eta_str}"
        header_padding = 58 - len(header_content)
        
        scanner_color = self._get_color('scanned')
        downloader_color = self._get_color('active')
        verifier_color = self._get_color('verifying')
        indexer_color = self._get_color('Indexed')
        aligning_color = self._get_color('aligning')
        cropping_color = self._get_color('cropping')
        yaml_writing_color = self._get_color('yaml_writing')
        
        return f"""
╔════════════════════════════════════════════════════════════╗
║ {header_content}{' ' * header_padding} ║
╚════════════════════════════════════════════════════════════╝
│ DataIngestor │ {scanner_color}Scanned: {self.counters['scanned']:3d}{self.RESET}
└──▼
│ FileScanner  │ Found: {self.counters['found']:3d} │ Missing: {self.counters['missing']:3d}
└──┬───────────┴─────────────┐
   │ (exists)        (missing)│
   ▼                          ▼
   │                  ┌───Downloader───┐ Queued: {self.counters['queued']:3d} │ {downloader_color}Active: {self.counters['active']:3d}{self.RESET} │ Finished: {self.counters['finished']:3d}
   │                  └───────┬────────┘
   │◄─────────────────────────┘
   ▼
│ Verifier     │ {verifier_color}Verifying: {self.counters['verifying']:3d}{self.RESET} │ OK: {self.counters['verified']:3d} │ Fail: {self.counters['failed']:3d} │ Err: {self.counters['errors']:3d}
└──┬────────────────┐
   │         (retry)│
   ▼                │
│ SceneIndexer │ {indexer_color}Indexed: {self.counters['Indexed']:3d}{self.RESET}
└──▼
│MetadataEnrich│ Enriching: {self.counters['enriching']:3d} │ Enriched: {self.counters['enriched']:3d}
└──▼
│AlignArtifact │ {aligning_color}Aligning: {self.counters['aligning']:3d}{self.RESET} │ Aligned: {self.counters['aligned']:3d}
└──▼
│ CropProducer │ {cropping_color}Cropping: {self.counters['cropping']:3d}{self.RESET} │ Cropped: {self.counters['cropped']:3d}
└──▼
│ YAMLArtifact │ {yaml_writing_color}Writing: {self.counters['yaml_writing']:3d}{self.RESET} │ Written: {self.counters['yaml_written']:3d}
└──▼
│FinalConsumer │ {self.BOLD}{self.GREEN}Complete: {self.counters['complete']:3d}{self.RESET}
"""

    async def update(self, **kwargs) -> None:
        """Update counters and refresh display."""
        async with self.lock:
            for key, value in kwargs.items():
                if key in self.counters:
                    self.counters[key] += value
            
            if 'active' in kwargs and kwargs['active'] > 0:
                if self.download_start_time is None:
                    self.download_start_time = trio.current_time()
                self.total_downloads += kwargs['active']
            
            self._display()

    def _display(self) -> None:
        """Display the current state to terminal."""
        if self._last_lines > 0:
            sys.stdout.write(f'\033[{self._last_lines}A')

        output = self._render()
        lines = output.count('\n')
        self._last_lines = lines

        sys.stdout.write(output)
        sys.stdout.flush()

    def clear(self) -> None:
        """Initial display."""
        self._display()
