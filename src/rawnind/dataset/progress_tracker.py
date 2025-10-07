"""
Progress tracking utilities for domain-specific tasks.
Extracted from mt_runner to provide clean separation of concerns.
"""

from typing import Any, Dict, Optional, Callable


class AlignmentProgressTracker:
    """
    Tracks progress for alignment computation tasks.
    Extracts scene and method information from results for display.
    """

    def __init__(self):
        self.current_scene: Optional[str] = None
        self.current_method: Optional[str] = None

    def extract_description(self, result: Any) -> Optional[str]:
        """
        Extract progress description from a result object.

        Args:
            result: Result object from alignment computation

        Returns:
            Description string if extractable, None otherwise
        """
        # Check if this is an alignment result with expected fields
        if not isinstance(result, dict):
            return None

        if "gt_fpath" not in result:
            return None

        # Extract scene and method info
        scene_name = result.get("image_set", "unknown")
        method = result.get("alignment_method", "unknown")

        # Only create new description if something changed
        if scene_name != self.current_scene or method != self.current_method:
            self.current_scene = scene_name
            self.current_method = method
            return f"Scene: {scene_name:<30} Method: {method.upper()}"

        return None


class GenericProgressTracker:
    """
    Generic progress tracker that can be customized with extraction functions.
    """

    def __init__(self,
                 extract_fn: Optional[Callable[[Any], Optional[str]]] = None,
                 default_desc: str = "Processing"):
        """
        Initialize generic progress tracker.

        Args:
            extract_fn: Function to extract description from result
            default_desc: Default description if extraction fails
        """
        self.extract_fn = extract_fn
        self.default_desc = default_desc
        self._last_desc: Optional[str] = None

    def extract_description(self, result: Any) -> Optional[str]:
        """
        Extract progress description from a result object.

        Args:
            result: Result object

        Returns:
            Description string if extractable or changed, None otherwise
        """
        if self.extract_fn:
            desc = self.extract_fn(result)
            if desc and desc != self._last_desc:
                self._last_desc = desc
                return desc
        return None

    def reset(self):
        """Reset tracker state."""
        self._last_desc = None


def create_tracker_for_task(task_type: str) -> Optional[GenericProgressTracker]:
    """
    Factory function to create appropriate progress tracker for task type.

    Args:
        task_type: Type of task ('alignment', 'enrichment', etc.)

    Returns:
        Appropriate progress tracker or None for generic tasks
    """
    if task_type == "alignment":
        tracker = AlignmentProgressTracker()
        return GenericProgressTracker(extract_fn=tracker.extract_description)
    elif task_type == "enrichment":
        # Define enrichment-specific extraction
        def extract_enrichment(result: Any) -> Optional[str]:
            if isinstance(result, dict) and "scene_name" in result:
                return f"Enriching scene: {result['scene_name']}"
            return None
        return GenericProgressTracker(extract_fn=extract_enrichment)
    else:
        # Return None for truly generic tasks
        return None