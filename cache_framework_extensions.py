"""
Extensions for the Keywords4CV cache framework.
Provides additional specialized cache implementations and utilities.
"""

import numpy as np
import torch
import logging
import gc
from typing import Dict, Any, Optional, Union, List, Set, Tuple, TypeVar, Generic, cast
import threading

from cache_framework import (
    CacheManager,
    VectorCacheManager,
    optimize_gpu_memory,
    IntegratedCacheSystem,
)
from cache_framework_types import (
    CacheStats,
    VectorType,
    VectorDict,
    SimilarityDict,
    SimilarityMatrix,
)

logger = logging.getLogger(__name__)


class MultiDimensionalVectorCache:
    """
    Enhanced vector cache that supports multiple vector dimensions.
    Useful when working with different embeddings models simultaneously.
    """

    def __init__(self, config: Dict[str, Any], nlp_models: Dict[str, Any]):
        """
        Initialize the multi-dimensional vector cache.

        Args:
            config: Configuration dictionary
            nlp_models: Dictionary mapping model names to NLP model instances
        """
        self.config = config
        self.nlp_models = nlp_models
        self.vector_caches: Dict[str, VectorCacheManager] = {}
        self._lock = threading.RLock()

        # Initialize vector caches for each model
        for model_name, model in nlp_models.items():
            self.vector_caches[model_name] = VectorCacheManager(
                config, model, namespace=f"vector_{model_name}"
            )

        logger.info(
            f"Initialized multi-dimensional vector cache with {len(nlp_models)} models"
        )

    def get_vector(self, term: str, model_name: str) -> Optional[VectorType]:
        """
        Get vector for a term using the specified model.

        Args:
            term: The term to vectorize
            model_name: Name of the model to use

        Returns:
            Vector if available, otherwise None
        """
        if model_name not in self.vector_caches:
            logger.warning(
                f"Model '{model_name}' not found in multi-dimensional vector cache"
            )
            return None

        return self.vector_caches[model_name].get_vector(term)

    def get_vectors_batch(
        self, terms: List[str], model_name: str, batch_size: int = 50
    ) -> VectorDict:
        """
        Get vectors for multiple terms using the specified model.

        Args:
            terms: List of terms to vectorize
            model_name: Name of the model to use
            batch_size: Size of batches for processing

        Returns:
            Dictionary mapping terms to vectors
        """
        if model_name not in self.vector_caches:
            logger.warning(
                f"Model '{model_name}' not found in multi-dimensional vector cache"
            )
            return {}

        return self.vector_caches[model_name].get_vectors_batch(terms, batch_size)

    def calculate_cross_model_similarity(
        self, term: str, candidates: List[str], source_model: str, target_model: str
    ) -> SimilarityDict:
        """
        Calculate similarity between a term in one model's embedding space and
        candidates in another model's embedding space.

        Args:
            term: The query term
            candidates: List of candidate terms
            source_model: Model name for the query term
            target_model: Model name for the candidates

        Returns:
            Dictionary mapping candidates to similarity scores
        """
        if (
            source_model not in self.vector_caches
            or target_model not in self.vector_caches
        ):
            logger.warning(
                f"One or both models not found: {source_model}, {target_model}"
            )
            return {}

        term_vector = self.vector_caches[source_model].get_vector(term)
        if term_vector is None:
            return {candidate: 0.0 for candidate in candidates}

        # Get vectors for all candidates using the target model
        candidate_vectors = self.vector_caches[target_model].get_vectors_batch(
            candidates
        )

        # Calculate similarities
        similarities = {}
        for candidate, vector in candidate_vectors.items():
            if vector is not None:
                # Ensure dimensions match - project to common space if needed
                if term_vector.shape != vector.shape:
                    # Use simple zero padding for dimension matching
                    if len(term_vector) < len(vector):
                        term_vector_padded = np.pad(
                            term_vector, (0, len(vector) - len(term_vector)), "constant"
                        )
                        sim = self._cosine_similarity(term_vector_padded, vector)
                    else:
                        vector_padded = np.pad(
                            vector, (0, len(term_vector) - len(vector)), "constant"
                        )
                        sim = self._cosine_similarity(term_vector, vector_padded)
                else:
                    sim = self._cosine_similarity(term_vector, vector)

                similarities[candidate] = max(0.0, min(1.0, sim))  # Clamp to [0,1]
            else:
                similarities[candidate] = 0.0

        return similarities

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 0 and norm2 > 0:
            return np.dot(vec1, vec2) / (norm1 * norm2)
        return 0.0

    def optimize_memory(self) -> Dict[str, int]:
        """
        Optimize memory usage across all vector caches.

        Returns:
            Dictionary with number of items trimmed per model
        """
        result = {}

        with self._lock:
            for model_name, cache in self.vector_caches.items():
                trimmed = cache.cache_manager.trim(20)  # Trim 20% from each cache
                result[model_name] = trimmed

            # Optimize GPU memory if available
            if any(cache.gpu_available for cache in self.vector_caches.values()):
                optimize_gpu_memory()

            # Force garbage collection
            gc.collect()

        total_trimmed = sum(result.values())
        if total_trimmed > 0:
            logger.info(
                f"Multi-dimensional vector cache: trimmed {total_trimmed} entries"
            )

        return result

    def get_stats(self) -> Dict[str, CacheStats]:
        """
        Get statistics for all vector caches.

        Returns:
            Dictionary mapping model names to their cache statistics
        """
        stats = {}
        for model_name, cache in self.vector_caches.items():
            stats[model_name] = cache.get_stats()
        return stats


class EnhancedIntegratedCacheSystem(IntegratedCacheSystem):
    """
    Enhanced version of the integrated cache system with support for multiple
    vector dimensionalities and advanced optimization strategies.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the enhanced integrated cache system.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.multi_vector_cache: Optional[MultiDimensionalVectorCache] = None

    def set_multi_models(self, nlp_models: Dict[str, Any]) -> None:
        """
        Set multiple NLP models for multi-dimensional vector caching.

        Args:
            nlp_models: Dictionary mapping model names to NLP model instances
        """
        with self._lock:
            self.multi_vector_cache = MultiDimensionalVectorCache(
                self.config, nlp_models
            )

    def get_vector_multi(self, term: str, model_name: str) -> Optional[VectorType]:
        """
        Get vector for a term using a specific model.

        Args:
            term: The term to vectorize
            model_name: Name of the model to use

        Returns:
            Vector if available, otherwise None
        """
        if not self.multi_vector_cache:
            logger.warning("Multi-dimensional vector cache not initialized")
            return None

        vector = self.multi_vector_cache.get_vector(term, model_name)

        with self._lock:
            if vector is not None:
                self.stats["vector_hits"] += 1
            else:
                self.stats["vector_misses"] += 1

        return vector

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage across all caches with enhanced support for
        multi-dimensional vector caches.

        Returns:
            Dictionary with optimization statistics
        """
        # Call the parent method to optimize standard caches
        basic_result = super().optimize_memory_usage()

        # Also optimize multi-dimensional vector cache if available
        multi_vector_result = {}
        if self.multi_vector_cache:
            multi_vector_result = self.multi_vector_cache.optimize_memory()

        # Combine results
        result = {"basic": basic_result, "multi_vector": multi_vector_result}

        return result


class OptimizedMultiDimensionalVectorCache(MultiDimensionalVectorCache):
    """
    Optimized version of MultiDimensionalVectorCache with better memory management
    and adaptive batch sizing for constrained environments.
    """

    def __init__(self, config: Dict[str, Any], nlp_models: Dict[str, Any]):
        """
        Initialize the optimized multi-dimensional vector cache.

        Args:
            config: Configuration dictionary
            nlp_models: Dictionary mapping model names to NLP model instances
        """
        super().__init__(config, nlp_models)

        # Enhanced memory management settings
        self.memory_check_interval = config.get("caching", {}).get(
            "memory_check_interval", 60
        )  # More frequent checks than parent class
        self.last_memory_check = time.time()
        self.max_memory_percent = config.get("hardware_limits", {}).get(
            "max_ram_usage_percent", 75
        )  # Lower threshold for constrained environments
        self.adaptive_batch_size = True

        # Determine vector dimensions for each model to estimate memory needs
        self.model_dimensions = {}
        for model_name, model in nlp_models.items():
            try:
                # Try to get vector dimension from model
                test_text = "test"
                test_vector = None

                if hasattr(model, "encode"):
                    # Try transformer-like encode method
                    test_vector = model.encode([test_text])[0]
                elif hasattr(model, "__call__"):
                    # Try spaCy-like call method
                    doc = model(test_text)
                    if hasattr(doc, "vector"):
                        test_vector = doc.vector

                if test_vector is not None:
                    self.model_dimensions[model_name] = len(test_vector)
                    logger.debug(
                        f"Model {model_name} vector dimension: {len(test_vector)}"
                    )
                else:
                    # Fallback value
                    self.model_dimensions[model_name] = 300
                    logger.warning(
                        f"Could not determine vector dimension for {model_name}, using default: 300"
                    )
            except Exception as e:
                logger.warning(
                    f"Error determining vector dimension for {model_name}: {e}"
                )
                self.model_dimensions[model_name] = 300

    def get_vectors_batch(
        self, terms: List[str], model_name: str, batch_size: int = 50
    ) -> VectorDict:
        """
        Get vectors for multiple terms with adaptive batch sizing based on available memory.

        Args:
            terms: List of terms to vectorize
            model_name: Name of the model to use
            batch_size: Starting batch size (will be adjusted based on available memory)

        Returns:
            Dictionary mapping terms to vectors
        """
        if model_name not in self.vector_caches:
            logger.warning(
                f"Model '{model_name}' not found in multi-dimensional vector cache"
            )
            return {}

        # Start with evaluating available memory
        try:
            import psutil

            available_mb = psutil.virtual_memory().available / (1024 * 1024)

            # Get vector dimension for this model
            vector_dim = self.model_dimensions.get(model_name, 300)

            # Calculate adaptive batch size based on available memory
            if self.adaptive_batch_size:
                from cache_framework_improvements import adaptive_batch_size_calculator

                adjusted_batch_size = adaptive_batch_size_calculator(
                    available_mb,
                    vector_dim,
                    safety_factor=0.4,  # Conservative for 8GB systems
                    min_batch=5,
                    max_batch=batch_size,  # Never exceed requested batch size
                )

                # Only log if we're significantly changing the batch size
                if abs(adjusted_batch_size - batch_size) > 5:
                    logger.debug(
                        f"Adjusted batch size from {batch_size} to {adjusted_batch_size} based on "
                        f"available memory ({available_mb:.1f}MB) and vector dimension ({vector_dim})"
                    )

                batch_size = adjusted_batch_size
        except ImportError:
            logger.debug("psutil not available, using default batch size")
        except Exception as e:
            logger.warning(f"Error calculating adaptive batch size: {e}")

        # Delegate to the appropriate vector cache with adjusted batch size
        return self.vector_caches[model_name].get_vectors_batch(terms, batch_size)

    def calculate_cross_model_similarity(
        self, term: str, candidates: List[str], source_model: str, target_model: str
    ) -> SimilarityDict:
        """
        Calculate similarity between a term in one model's embedding space and
        candidates in another model's embedding space, with improved memory efficiency.

        Args:
            term: The query term
            candidates: List of candidate terms
            source_model: Model name for the query term
            target_model: Model name for the candidates

        Returns:
            Dictionary mapping candidates to similarity scores
        """
        if (
            source_model not in self.vector_caches
            or target_model not in self.vector_caches
        ):
            logger.warning(
                f"One or both models not found: {source_model}, {target_model}"
            )
            return {}

        # Memory-efficient similarity calculation
        try:
            term_vector = self.vector_caches[source_model].get_vector(term)
            if term_vector is None:
                return {candidate: 0.0 for candidate in candidates}

            # Get available memory before batch operation
            try:
                import psutil

                available_mb = psutil.virtual_memory().available / (1024 * 1024)

                # Adjust batch size based on available memory
                if available_mb < 1000:  # Less than 1GB available
                    batch_size = max(5, min(20, len(candidates) // 5))
                    logger.debug(
                        f"Low memory ({available_mb:.1f}MB), using smaller batch size: {batch_size}"
                    )
                else:
                    batch_size = min(50, len(candidates))
            except Exception:
                # Default batch size if memory check fails
                batch_size = min(50, len(candidates))

            # Process in batches to limit memory usage
            similarities = {}
            for i in range(0, len(candidates), batch_size):
                batch_candidates = candidates[i : i + batch_size]

                # Get vectors for this batch
                candidate_vectors = self.vector_caches[target_model].get_vectors_batch(
                    batch_candidates, batch_size
                )

                # Calculate similarities for this batch
                for candidate, vector in candidate_vectors.items():
                    if vector is not None:
                        # Use memory-efficient similarity calculation
                        try:
                            from cache_framework_improvements import (
                                memory_efficient_similarity,
                            )

                            # Handle dimension mismatch
                            if len(term_vector) != len(vector):
                                # Use simple zero padding
                                if len(term_vector) < len(vector):
                                    term_vector_padded = np.pad(
                                        term_vector,
                                        (0, len(vector) - len(term_vector)),
                                        "constant",
                                    )
                                    sim = memory_efficient_similarity(
                                        term_vector_padded, vector
                                    )
                                else:
                                    vector_padded = np.pad(
                                        vector,
                                        (0, len(term_vector) - len(vector)),
                                        "constant",
                                    )
                                    sim = memory_efficient_similarity(
                                        term_vector, vector_padded
                                    )
                            else:
                                sim = memory_efficient_similarity(term_vector, vector)
                        except ImportError:
                            # Fall back to original method
                            sim = self._cosine_similarity(term_vector, vector)

                        similarities[candidate] = max(
                            0.0, min(1.0, float(sim))
                        )  # Clamp to [0,1]
                    else:
                        similarities[candidate] = 0.0

                # Check memory after each batch and force GC if needed
                self._check_memory_after_batch()

            return similarities

        except Exception as e:
            logger.error(f"Error calculating cross-model similarity: {e}")
            return {candidate: 0.0 for candidate in candidates}

    def _check_memory_after_batch(self):
        """Check memory usage after batch operations and optimize if needed."""
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return  # Not time to check yet

        try:
            import psutil

            memory_usage = psutil.virtual_memory().percent
            self.last_memory_check = current_time

            if memory_usage > self.max_memory_percent:
                logger.warning(f"High memory usage after vector batch: {memory_usage}%")
                # Optimize memory more aggressively
                self.optimize_memory()

                # Force garbage collection
                import gc

                gc.collect()

                # Clear CUDA cache if available
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared CUDA cache to reduce memory pressure")
                except ImportError:
                    pass
        except Exception as e:
            logger.warning(f"Error checking memory usage: {e}")


class MemoryConstrainedCacheSystem(EnhancedIntegratedCacheSystem):
    """
    Cache system optimized for memory-constrained environments (like laptops with 8GB RAM).
    Provides more aggressive memory management and efficient vector operations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the memory-constrained cache system.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # More aggressive memory management
        memory_config = config.get("hardware_limits", {})
        if "max_ram_usage_percent" not in memory_config:
            # Lower the default threshold for constrained environments
            memory_config["max_ram_usage_percent"] = 70

        # Use optimized vector cache by default
        self.optimized_multi_vector_cache = None

        # Serialization settings
        self.use_msgpack = config.get("caching", {}).get("use_msgpack", True)

        # Configure disk cache with msgpack if available
        if config.get("caching", {}).get("cache_dir") and self.use_msgpack:
            try:
                from cache_framework_improvements import (
                    configure_disk_cache_serialization,
                )

                cache_dir = config.get("caching", {}).get("cache_dir")

                # Apply msgpack configuration to disk cache if possible
                if hasattr(self.base_cache, "backend") and hasattr(
                    self.base_cache.backend, "disk_cache"
                ):
                    if self.base_cache.backend.disk_cache is not None:
                        disk_config = configure_disk_cache_serialization(
                            cache_dir, use_msgpack=True
                        )

                        # Apply serialization functions if available
                        if "serialize" in disk_config and "deserialize" in disk_config:
                            self.base_cache.backend.disk_cache.serialize = disk_config[
                                "serialize"
                            ]
                            self.base_cache.backend.disk_cache.deserialize = (
                                disk_config["deserialize"]
                            )
                            logger.info("Applied msgpack serialization to disk cache")
            except Exception as e:
                logger.warning(f"Could not configure msgpack serialization: {e}")

        # Initialize memory manager for coordinated memory management
        try:
            from memory_manager import create_memory_manager

            self.memory_manager = create_memory_manager(config)

            # Register caches with the memory manager
            if self.base_cache and hasattr(self.base_cache, "backend"):
                self.memory_manager.register_component(self.base_cache.backend)

            # Register text caches
            if hasattr(self, "text_cache"):
                self.memory_manager.register_component(
                    self.text_cache.preprocess_cache.backend
                )
                self.memory_manager.register_component(
                    self.text_cache.tokenize_cache.backend
                )
                self.memory_manager.register_component(
                    self.text_cache.ngram_cache.backend
                )
                self.memory_manager.register_component(
                    self.text_cache.sentence_cache.backend
                )

            logger.info("Memory manager initialized for coordinated cache management")
        except ImportError:
            logger.warning(
                "Memory manager module not available, falling back to basic memory management"
            )
            self.memory_manager = None

        # Use optimized vector similarity function when available
        try:
            from cache_framework_improvements import (
                optimized_similarity_for_constrained_memory,
            )

            self.optimized_similarity = optimized_similarity_for_constrained_memory
            logger.info("Using memory-optimized similarity function")
        except (ImportError, AttributeError):
            self.optimized_similarity = None
            logger.info("Memory-optimized similarity function not available")

    def set_multi_models(self, nlp_models: Dict[str, Any]) -> None:
        """
        Set multiple NLP models with optimized vector caching for memory efficiency.
        Properly integrates with memory manager for coordinated memory management.
        """
        with self._lock:
            # Use our optimized vector cache implementation
            self.optimized_multi_vector_cache = OptimizedMultiDimensionalVectorCache(
                self.config, nlp_models
            )

            # Also set in parent class for compatibility
            self.multi_vector_cache = self.optimized_multi_vector_cache

            # Register with memory manager if available
            if hasattr(self, "memory_manager") and self.memory_manager is not None:
                for (
                    model_name,
                    cache,
                ) in self.optimized_multi_vector_cache.vector_caches.items():
                    self.memory_manager.register_component(cache.cache_manager)
                    logger.debug(
                        f"Registered vector cache for model {model_name} with memory manager"
                    )

        logger.info(
            f"Initialized optimized multi-vector cache with {len(nlp_models)} models"
        )

    def get_vector_multi(self, term: str, model_name: str) -> Optional[VectorType]:
        """
        Get vector for a term using a specific model, with memory optimization.

        Args:
            term: The term to vectorize
            model_name: Name of the model to use

        Returns:
            Vector if available, otherwise None
        """
        # Use optimized cache if available
        if self.optimized_multi_vector_cache:
            result = self.optimized_multi_vector_cache.get_vector(term, model_name)

            with self._lock:
                if result is not None:
                    self.stats["vector_hits"] += 1
                else:
                    self.stats["vector_misses"] += 1

            return result

        # Fall back to parent implementation
        return super().get_vector_multi(term, model_name)

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage with more aggressive settings for memory-constrained environments.
        Uses memory manager for coordinated optimization when available.
        """
        # If we have a memory manager, let it handle coordinated optimization
        if hasattr(self, "memory_manager") and self.memory_manager is not None:
            # Still get baseline results from parent
            result = super().optimize_memory_usage()

            # Memory manager will handle coordination across components
            # Just need to trigger severe cleanup in extreme cases
            try:
                memory_status = self.memory_manager.get_memory_status()
                if memory_status.get("percent", 0) > memory_status.get(
                    "emergency_threshold", 85
                ):
                    # Emergency situation - very aggressive trimming
                    # Force GC and GPU memory cleanup
                    import gc

                    gc.collect(generation=2)  # Force full collection

                    # Optimize GPU memory if available
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.info("Emergency cleanup: cleared CUDA cache")
                    except ImportError:
                        pass
            except Exception as e:
                logger.error(f"Error during emergency memory optimization: {e}")

            return result
        else:
            # Fall back to parent implementation
            return super().optimize_memory_usage()

    def calculate_similarity_batch(
        self, term: str, candidates: List[str]
    ) -> SimilarityDict:
        """Calculate similarity with memory optimization."""
        # ...existing code...

        # Use optimized similarity function if available
        if (
            hasattr(self, "optimized_similarity")
            and self.optimized_similarity is not None
        ):
            try:
                # Use lower precision for memory constrained environments
                return self.optimized_similarity(
                    term_vector, vector, precision="float16"
                )
            except Exception as e:
                logger.warning(f"Optimized similarity failed, falling back: {e}")

        # Fall back to regular implementation
        # ...existing code...


# Example usage:
"""
# Initialize enhanced cache system
config = {...}  # Your configuration
cache_system = EnhancedIntegratedCacheSystem(config)

# Set up multiple NLP models
models = {
    "spacy_lg": spacy.load("en_core_web_lg"),
    "distilbert": transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
}
cache_system.set_multi_models(models)

# Use specific model for vectorization
vector = cache_system.get_vector_multi("example", "spacy_lg")

# Compare across models
similarities = cache_system.multi_vector_cache.calculate_cross_model_similarity(
    "example", ["test", "sample"], "spacy_lg", "distilbert"
)
"""
