"""
Cache invalidation utilities for Keywords4CV.
Provides specialized functions for efficient and selective cache invalidation.
"""

import logging
import json
import xxhash
import re
from typing import Dict, Any, List, Set, Optional, Union, Tuple
import threading
from cache_framework import CACHE_VERSION

# Setup logging
logger = logging.getLogger(__name__)


class ConfigChangeDetector:
    """
    Detects changes in configuration to determine which cache sections need invalidation.
    Uses smart diffing to identify relevant changes.
    """

    def __init__(self):
        """Initialize the change detector."""
        # Define section mappings to cache types
        self.section_to_cache_map = {
            "stop_words": ["text", "tokenization", "preprocessing"],
            "stop_words_add": ["text", "tokenization", "preprocessing"],
            "stop_words_exclude": ["text", "tokenization", "preprocessing"],
            "text_processing": ["text", "tokenization", "preprocessing", "ngram"],
            "caching": ["metadata", "configuration"],
            "vectorization": ["vector", "embedding"],
            "validation": ["validation", "metrics"],
            "keyword_categories": ["extraction", "keywords"],
            "hardware_limits": ["vector", "performance"],
            "nlp": ["vector", "embedding", "tokenization"],
        }

        # Specific keys that require full invalidation if changed
        self.critical_keys = {
            "model_type",
            "model_name",
            "cache_version",
            "embedding_dimension",
            "token_pattern",
            "vector_library",
        }

    def detect_changes(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> Tuple[Set[str], Set[str]]:
        """
        Detect important changes in configuration.

        Args:
            old_config: Previous configuration
            new_config: New configuration

        Returns:
            Tuple of (changed_sections, affected_cache_types)
        """
        changed_sections = set()
        affected_cache_types = set()

        # Check for critical key changes anywhere in the config
        if self._has_critical_changes(old_config, new_config):
            # Signal that all caches need invalidation
            return set(self.section_to_cache_map.keys()), {"all"}

        # Compare each section
        for section in set(old_config.keys()) | set(new_config.keys()):
            old_section = old_config.get(section, {})
            new_section = new_config.get(section, {})

            if old_section != new_section:
                # Compute detailed hash to avoid unnecessary invalidation
                old_hash = xxhash.xxh3_64(
                    json.dumps(old_section, sort_keys=True).encode()
                ).hexdigest()
                new_hash = xxhash.xxh3_64(
                    json.dumps(new_section, sort_keys=True).encode()
                ).hexdigest()

                if old_hash != new_hash:
                    changed_sections.add(section)
                    # Map section to affected cache types
                    affected_types = self.section_to_cache_map.get(section, [])
                    affected_cache_types.update(affected_types)

        return changed_sections, affected_cache_types

    def _has_critical_changes(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> bool:
        """
        Check if any critical keys have changed that would require full invalidation.

        Args:
            old_config: Previous configuration
            new_config: New configuration

        Returns:
            bool: True if critical changes detected
        """
        # Flatten configs to check nested keys
        old_flat = self._flatten_dict(old_config)
        new_flat = self._flatten_dict(new_config)

        # Check critical keys
        for key in self.critical_keys:
            # Look for the key anywhere in the flattened dict
            for flat_key, old_val in old_flat.items():
                if key in flat_key:
                    # Check if this key exists and has changed in new config
                    if flat_key in new_flat and old_val != new_flat[flat_key]:
                        logger.info(f"Critical config change detected in {flat_key}")
                        return True

        return False

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
        """
        Flatten a nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items

        Returns:
            Dict[str, Any]: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)


class InvalidationRegistry:
    """
    Registry that tracks which cache segments need to be invalidated and when.
    Provides dependency resolution for inter-related caches.
    """

    def __init__(self):
        """Initialize the registry."""
        # Mapping from cache segments to their invalidation timestamps
        self.invalidation_timestamps = {}

        # Dependencies between cache segments
        self.dependencies = {
            "tokenization": ["preprocessing"],  # Tokenization depends on preprocessing
            "ngram": ["tokenization"],  # N-grams depend on tokenization
            "vector": ["tokenization"],  # Vectors depend on tokenization
            "keywords": ["vector", "ngram"],  # Keywords depend on vectors and n-grams
            "validation": ["keywords"],  # Validation depends on keywords
        }

        self._lock = threading.RLock()

    def mark_invalidated(self, segment: str) -> None:
        """
        Mark a cache segment as invalidated with current timestamp.

        Args:
            segment: Cache segment name
        """
        with self._lock:
            import time

            self.invalidation_timestamps[segment] = time.time()

    def is_valid(self, segment: str, timestamp: float) -> bool:
        """
        Check if a cache entry is still valid compared to invalidation timestamp.

        Args:
            segment: Cache segment name
            timestamp: Cache entry creation timestamp

        Returns:
            bool: True if the cache entry is still valid
        """
        with self._lock:
            # Get invalidation timestamp for this segment
            invalidation_time = self.invalidation_timestamps.get(segment, 0)

            # Check if invalidated after the cache entry was created
            if invalidation_time > timestamp:
                return False

            # Also check all dependencies
            for dependency in self.get_dependencies(segment):
                dep_invalidation_time = self.invalidation_timestamps.get(dependency, 0)
                if dep_invalidation_time > timestamp:
                    return False

            return True

    def get_dependencies(self, segment: str) -> Set[str]:
        """
        Get all dependencies for a cache segment (direct and transitive).

        Args:
            segment: Cache segment name

        Returns:
            Set[str]: Set of all dependencies
        """
        result = set()

        def collect_deps(seg):
            direct_deps = self.dependencies.get(seg, [])
            for dep in direct_deps:
                if dep not in result:
                    result.add(dep)
                    collect_deps(dep)

        collect_deps(segment)
        return result


def analyze_config_for_invalidation(
    old_config: Dict[str, Any], new_config: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    Analyze configuration changes and determine which cache types need invalidation.

    Args:
        old_config: Previous configuration
        new_config: New configuration

    Returns:
        Dict mapping cache systems to lists of cache types that need invalidation
    """
    detector = ConfigChangeDetector()
    changed_sections, affected_types = detector.detect_changes(old_config, new_config)

    # Organize by cache system
    result = {
        "vector_cache": [],
        "text_cache": [],
        "general_cache": [],
        "integrated_cache": [],
    }

    # Special handling for "all" which means everything needs invalidation
    if "all" in affected_types:
        return {
            "vector_cache": ["all"],
            "text_cache": ["all"],
            "general_cache": ["all"],
            "integrated_cache": ["all"],
        }

    # Map affected types to specific cache systems
    for affected_type in affected_types:
        if affected_type in ["vector", "embedding"]:
            result["vector_cache"].append(affected_type)
        elif affected_type in ["text", "tokenization", "preprocessing", "ngram"]:
            result["text_cache"].append(affected_type)
        elif affected_type in ["metadata", "configuration"]:
            result["general_cache"].append(affected_type)

        # Always include in integrated cache to ensure central tracking
        result["integrated_cache"].append(affected_type)

    # Log detailed analysis results
    if changed_sections:
        change_list = ", ".join(sorted(changed_sections))
        affected_list = ", ".join(sorted(affected_types))
        logger.info(
            f"Config changes in: {change_list} affect cache types: {affected_list}"
        )

    return result


def get_cache_invalidation_marker(component_name: str, config: Dict[str, Any]) -> str:
    """
    Generate a unique marker for cache invalidation that can be stored with cache entries.

    Args:
        component_name: Name of the cache component
        config: Configuration dictionary

    Returns:
        str: Unique invalidation marker
    """
    # Extract only the most relevant parts of config to generate the marker
    relevant_sections = [
        "stop_words",
        "text_processing",
        "vectorization",
        "caching",
        "hardware_limits",
    ]

    # Build a minimal config with only the relevant sections
    minimal_config = {}
    for section in relevant_sections:
        if section in config:
            minimal_config[section] = config[section]

    # Add cache version if available
    if "cache_version" in config.get("caching", {}):
        minimal_config["version"] = config["caching"]["cache_version"]
    elif "caching" in minimal_config and isinstance(minimal_config["caching"], dict):
        minimal_config["caching"]["hash_version"] = (
            CACHE_VERSION  # Import from cache_framework
        )

    # Generate hash from the minimal config
    config_str = json.dumps(minimal_config, sort_keys=True)
    return f"{component_name}_{xxhash.xxh3_64(config_str.encode()).hexdigest()[:16]}"
