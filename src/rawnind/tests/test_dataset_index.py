"""Tests for DatasetIndex event-based cache invalidation system."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from rawnind.dataset.manager import DatasetIndex, CacheEvent


class TestEventBasedCacheInvalidation:
    """Test suite for event-based cache invalidation infrastructure."""

    def test_event_emission_on_index_structure_change(self):
        """Test that rebuilding index emits appropriate event.
        
        Given: A DatasetIndex instance
        When: _build_index_from_data() is called
        Then: INDEX_STRUCTURE_CHANGED event should be emitted
        """
        index = DatasetIndex()
        
        event_fired = False
        
        def listener():
            nonlocal event_fired
            event_fired = True
        
        index._events.on(CacheEvent.INDEX_STRUCTURE_CHANGED, listener)
        
        # Create minimal dataset data
        dataset_data = {
            'Bayer': {
                'test_scene': {
                    'clean_images': [{'filename': 'test.arw', 'sha1': 'abc123'}],
                    'noisy_images': [],
                    'unknown_sensor': False,
                    'test_reserve': False
                }
            }
        }
        
        # Load index (triggers _build_index_from_data)
        index._build_index_from_data(dataset_data)
        
        assert event_fired, "INDEX_STRUCTURE_CHANGED event should be emitted"

    def test_cache_invalidation_on_event(self):
        """Test that caches invalidate when relevant events fire.
        
        Given: A cached value that depends on specific events
        When: The relevant event is emitted
        Then: The cache should be invalidated
        """
        index = DatasetIndex()
        
        # Create minimal dataset data
        dataset_data = {
            'Bayer': {
                'test_scene': {
                    'clean_images': [{'filename': 'test.arw', 'sha1': 'abc123'}],
                    'noisy_images': [],
                    'unknown_sensor': False,
                    'test_reserve': False
                }
            }
        }
        
        index._build_index_from_data(dataset_data)
        
        # Prime cache by accessing known_extensions
        # Since no local_path is set, it will be empty but cached
        _ = index.known_extensions
        assert index._known_extensions is not None, "Cache should be populated"
        
        # Emit event that should invalidate this cache
        index._emit(CacheEvent.LOCAL_PATHS_UPDATED)
        
        # Cache should be invalidated
        assert index._known_extensions is None, "Cache should be invalidated after LOCAL_PATHS_UPDATED"

    def test_targeted_cache_invalidation(self):
        """Test that only affected caches are invalidated.
        
        Given: Multiple cached values with different dependencies
        When: A specific event is emitted
        Then: Only caches dependent on that event should be invalidated
        """
        index = DatasetIndex()
        
        # Create minimal dataset data
        dataset_data = {
            'Bayer': {
                'test_scene': {
                    'clean_images': [{'filename': 'test.arw', 'sha1': 'abc123'}],
                    'noisy_images': [],
                    'unknown_sensor': False,
                    'test_reserve': False
                }
            }
        }
        
        index._build_index_from_data(dataset_data)
        
        # Prime both caches
        extensions = index.known_extensions
        cfa_types = index.sorted_cfa_types
        
        assert index._known_extensions is not None, "known_extensions should be cached"
        assert index._sorted_cfa_types is not None, "sorted_cfa_types should be cached"
        
        # Emit LOCAL_PATHS_UPDATED (should not affect sorted_cfa_types)
        index._emit(CacheEvent.LOCAL_PATHS_UPDATED)
        
        # known_extensions should be invalidated
        assert index._known_extensions is None, "known_extensions should be invalidated"
        
        # sorted_cfa_types should still be cached
        assert index._sorted_cfa_types is not None, "sorted_cfa_types should remain cached"
        assert index._sorted_cfa_types == cfa_types, "sorted_cfa_types value should be unchanged"

    def test_index_structure_changed_invalidates_all(self):
        """Test that INDEX_STRUCTURE_CHANGED invalidates all dependent caches.
        
        Given: Multiple cached values
        When: INDEX_STRUCTURE_CHANGED event is emitted
        Then: Both known_extensions and sorted_cfa_types should be invalidated
        """
        index = DatasetIndex()
        
        # Create minimal dataset data
        dataset_data = {
            'Bayer': {
                'test_scene': {
                    'clean_images': [{'filename': 'test.arw', 'sha1': 'abc123'}],
                    'noisy_images': [],
                    'unknown_sensor': False,
                    'test_reserve': False
                }
            }
        }
        
        index._build_index_from_data(dataset_data)
        
        # Prime both caches
        _ = index.known_extensions
        _ = index.sorted_cfa_types
        
        assert index._known_extensions is not None
        assert index._sorted_cfa_types is not None
        
        # Emit INDEX_STRUCTURE_CHANGED
        index._emit(CacheEvent.INDEX_STRUCTURE_CHANGED)
        
        # Both should be invalidated
        assert index._known_extensions is None, "known_extensions should be invalidated"
        assert index._sorted_cfa_types is None, "sorted_cfa_types should be invalidated"

    def test_event_listener_registration_and_removal(self):
        """Test that event listeners can be registered and removed."""
        index = DatasetIndex()
        
        call_count = 0
        
        def listener():
            nonlocal call_count
            call_count += 1
        
        # Register listener
        index._events.on(CacheEvent.LOCAL_PATHS_UPDATED, listener)
        
        # Emit event
        index._emit(CacheEvent.LOCAL_PATHS_UPDATED)
        assert call_count == 1, "Listener should be called once"
        
        # Remove listener
        index._events.off(CacheEvent.LOCAL_PATHS_UPDATED, listener)
        
        # Emit event again
        index._emit(CacheEvent.LOCAL_PATHS_UPDATED)
        assert call_count == 1, "Listener should not be called after removal"

    def test_multiple_listeners_for_same_event(self):
        """Test that multiple listeners can be registered for the same event."""
        index = DatasetIndex()
        
        listener1_called = False
        listener2_called = False
        
        def listener1():
            nonlocal listener1_called
            listener1_called = True
        
        def listener2():
            nonlocal listener2_called
            listener2_called = True
        
        # Register both listeners
        index._events.on(CacheEvent.LOCAL_PATHS_UPDATED, listener1)
        index._events.on(CacheEvent.LOCAL_PATHS_UPDATED, listener2)
        
        # Emit event
        index._emit(CacheEvent.LOCAL_PATHS_UPDATED)
        
        assert listener1_called, "First listener should be called"
        assert listener2_called, "Second listener should be called"