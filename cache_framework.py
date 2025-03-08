"""
Keywords4CV Unified Cache Framework
====================================
version 0.1.0

This module provides a flexible and extensible caching system designed to optimize performance
in the Keywords4CV project. It supports multiple cache backends, specialized cache managers,
and integrates with security features to protect sensitive data.

Key Features:
-------------
- **Modular Design**: Easily switch between different cache backends (memory-only, hybrid disk+memory).
- **Specialized Cache Managers**: Optimized cache managers for vector embeddings, text processing, and general data.
- **Automatic Cache Invalidation**: Uses configuration hashing to automatically invalidate cache entries when settings change.
- **Memory Management**: Adaptive cache sizing and memory monitoring to prevent out-of-memory errors.
- **Efficient Serialization**: Intelligent format selection for different data types with SerializationManager.
- **Security**: Strong encryption for cached data, secure directory permissions, and memory wiping for sensitive information.
- **Thread Safety**: Coordinated locking mechanisms to ensure thread safety in concurrent environments.
- **Extensibility**: Designed to be easily extended with new cache backends and specialized cache managers.

Cache Backends:
---------------
- **MemoryCacheBackend**: Stores data in memory using LRU or TTLCache from the `cachetools` library.
- **HybridCacheBackend**: Combines an in-memory cache with a disk-based cache for larger datasets.
  Uses the `diskcache` library for persistent storage with efficient serialization.
- **EnhancedHybridCacheBackend**: Extends the hybrid cache with access counting and stronger security features.
- **LargeObjectCacheBackend**: Optimized for caching large objects using weak references to minimize memory footprint.

Cache Managers:
---------------
- **CacheManager**: Base class for managing cache operations. Provides a simple interface for
  getting, setting, clearing, and checking cache entries.
- **VectorCacheManager**: Specialized cache manager for storing and retrieving vector embeddings.
  Includes GPU acceleration and memory optimization techniques.
- **TextProcessingCacheManager**: Manages caches for text preprocessing, tokenization, and n-gram generation.
- **IntegratedCacheSystem**: Coordinates between specialized cache managers to provide a unified caching interface.

Serialization:
--------------
- **SerializationManager**: Central component that optimizes serialization based on data type:
  - Specialized handling for NumPy arrays and PyTorch tensors
  - Format selection from msgpack, flexbuffers, raw binary, or JSON
  - Keeps raw objects in memory while using compressed formats for disk storage
  - Optional encryption for sensitive data

Security Features:
------------------
- **Encryption**: Integrates with the `cache_framework_security` module to encrypt cached data using AES-GCM.
- **Secure Key Derivation**: Uses Argon2id (or PBKDF2 as a fallback) to derive encryption keys from a salt.
- **Secure Directory Handling**: Applies proper permissions to cache directories based on OS.
- **Secure Memory Handling**: Provides functions for securely wiping sensitive data from memory.

Memory Management:
------------------
- **Adaptive Cache Sizing**: Automatically adjusts cache sizes based on available memory and configuration settings.
- **Memory Monitoring**: Monitors memory usage and triggers cache trimming to prevent out-of-memory errors.
- **Batch Processing**: Optimized batch operations for vector calculations with dynamic batch sizing.
- **GPU Memory Optimization**: Provides functions for clearing unused GPU memory and cache.

Thread Safety:
--------------
- **Coordinated Locking**: Uses a coordinated lock management system to prevent deadlocks in concurrent environments.
- **Lock Hierarchy**: Defines a lock hierarchy to ensure that locks are acquired in a consistent order.
- **Minimized Lock Scope**: Critical sections are kept as small as possible to maximize concurrency.
"""

import json
import os
import xxhash
import psutil
import time
import logging
import threading
import gc
import numpy as np
import torch
import re
from typing import (
    Dict,
    Any,
    Optional,
    Union,
    List,
    Set,
    Tuple,
    TypeVar,
    Generic,
    cast,
    Callable,
    Type,
    Literal,
    TYPE_CHECKING,
    ContextManager,
)
from contextlib import contextmanager
from cachetools import LRUCache, TTLCache
from diskcache import Cache
import weakref
from threading import RLock
import importlib.util

# Set up logger before using it
logger = logging.getLogger(__name__)

# Import security module
try:
    from cache_framework_security import SecureCacheSerializer

    SECURITY_MODULE_AVAILABLE = True
except ImportError:
    SECURITY_MODULE_AVAILABLE = False
    SecureCacheSerializer = None
    logger.warning(
        "Cache security module not available. Cache data will not be encrypted."
    )

# Import the type definitions for improved type safety using importlib to check module existence
if importlib.util.find_spec("cache_framework_types"):
    from cache_framework_types import (
        CacheStats,
        VectorType,
        VectorDict,
        SimilarityDict,
        SimilarityMatrix,
        MemoryOptimizationResult,
        CacheKeyInfo,
        VectorProcessCallback,
    )
else:
    # Define fallback types if the module is not available
    CacheStats = Dict[str, Any]
    VectorType = Any
    VectorDict = Dict[str, Any]
    SimilarityDict = Dict[str, float]
    SimilarityMatrix = Any
    MemoryOptimizationResult = Dict[str, int]
    CacheKeyInfo = Dict[str, Any]
    VectorProcessCallback = Any
    logger.warning(
        "cache_framework_types module not found, using fallback type definitions"
    )

# Try to import lock_utils for improved lock management
try:
    from lock_utils import create_component_lock, coordinated_lock

    LOCK_UTILS_AVAILABLE = True
except ImportError:
    LOCK_UTILS_AVAILABLE = False

# Cache version for invalidation
CACHE_VERSION = "1.1.0"

# Type variables for generic typing support
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def get_cache_salt(config: Dict[str, Any]) -> str:
    """
    Retrieves the cache salt, prioritizing environment variables, then config, then a default.

    Args:
        config: The configuration dictionary

    Returns:
        str: The cache salt to use for hashing operations
    """
    return os.environ.get(
        "K4CV_CACHE_SALT",
        config.get("caching", {}).get("cache_salt", "default_secret_salt"),
    )


def calculate_optimal_cache_size(
    config: Dict[str, Any], cache_type: str = "default"
) -> int:
    """
    Calculate the optimal cache size based on available memory and configuration.

    Args:
        config: The configuration dictionary
        cache_type: The type of cache (default, vector, text, etc.)

    Returns:
        int: The calculated optimal cache size
    """
    # Get base cache size from config with fallbacks based on cache type
    type_config = config.get("caching", {}).get(cache_type, {})
    base_cache_size = type_config.get(
        "cache_size", config.get("caching", {}).get("cache_size", 5000)
    )

    # Get memory scaling factor for adaptive sizing
    scaling_factor = config.get("hardware_limits", {}).get("memory_scaling_factor", 0.3)

    # Get max cache size (upper limit)
    max_cache_size = type_config.get(
        "max_cache_size",
        config.get("caching", {}).get("max_cache_size", base_cache_size * 5),
    )

    if scaling_factor:
        # Calculate dynamic size based on available memory
        try:
            available_mb = psutil.virtual_memory().available / (1024 * 1024)

            # Scale by cache type - vectors need more memory per item
            type_multiplier = {
                "vector": 0.5,  # Vectors need more memory per item
                "text": 1.2,  # Text processing can use more items
                "default": 1.0,
            }.get(cache_type, 1.0)

            dynamic_size = int((available_mb * type_multiplier) / scaling_factor)

            # Apply reasonable bounds:
            # - Use at least the base size
            # - Don't exceed the maximum cache size
            # - Ensure we have at least some minimum amount of free memory
            minimum_free_mb = config.get("hardware_limits", {}).get(
                "minimum_free_mb", 500
            )

            if available_mb < minimum_free_mb * 2:
                # Low memory situation, be conservative
                return max(100, base_cache_size // 2)

            return max(base_cache_size, min(dynamic_size, max_cache_size))
        except Exception as e:
            logger.warning(f"Error calculating dynamic cache size: {e}")

    return base_cache_size


def hash_config_sections(
    config: Dict[str, Any], salt: str, sections: Optional[List[str]] = None
) -> str:
    """
    Create a hash of relevant configuration sections for cache invalidation.

    Args:
        config: Configuration dictionary
        salt: Salt value for the hash
        sections: Specific sections to include (if None, includes commonly cached sections)

    Returns:
        str: Hexadecimal hash of the configuration
    """
    if sections is None:
        sections = [
            "stop_words",
            "stop_words_add",
            "stop_words_exclude",
            "text_processing",
            "caching",
            "validation",
            "keyword_categories",
        ]

    relevant_config = {}
    for section in sections:
        if section in config:
            relevant_config[section] = config.get(section)

    config_str = json.dumps(relevant_config, sort_keys=True)
    return xxhash.xxh3_64(f"{salt}_{config_str}".encode("utf-8")).hexdigest()


def check_memory_usage(
    component_id: str,
    last_check_time: float,
    check_interval: float,
    memory_threshold: Optional[float] = None,
    trim_callback: Optional[Callable[[float], None]] = None,
) -> float:
    """
    Centralized memory monitoring function that respects memory manager coordination.
    Uses lock_utils if available to avoid lock conflicts.

    Args:
        component_id: Unique identifier for the component
        last_check_time: Timestamp of the last memory check
        check_interval: Minimum time between checks in seconds
        memory_threshold: Memory usage threshold to trigger trimming (percentage)
        trim_callback: Optional callback function to trim memory when threshold is exceeded

    Returns:
        float: Updated last_check_time
    """
    # Check if enough time has passed since the last check
    current_time = time.time()
    if current_time - last_check_time < check_interval:
        return last_check_time

    # First try to get the memory manager directly
    try:
        from memory_utils import get_memory_manager

        memory_manager = get_memory_manager()

        if memory_manager is not None:
            # If we have a memory manager, let it decide if checking is necessary
            if not memory_manager.should_check_memory(component_id):
                return last_check_time

            # Check with memory manager to see if we should optimize
            if memory_threshold is not None and trim_callback is not None:
                should_optimize, trim_percent = memory_manager.component_memory_check(
                    component_id, psutil.virtual_memory().percent
                )
                if should_optimize:
                    try:
                        trim_callback(psutil.virtual_memory().percent)
                    except Exception as e:
                        logger.warning(
                            f"Memory trim callback error for {component_id}: {e}"
                        )

            return current_time
    except ImportError:
        # Continue to other approaches if direct manager access fails
        pass

    # Try to use memory_utils.check_memory_usage if available
    try:
        from memory_utils import check_memory_usage as memory_utils_check

        try:
            # Try with all parameters (for newer versions of memory_utils)
            return memory_utils_check(
                component_id,
                last_check_time,
                check_interval,
                memory_threshold,
                trim_callback,
            )
        except TypeError:
            # Fall back to basic signature if external function doesn't accept all parameters
            result = memory_utils_check(component_id, last_check_time, check_interval)

            # If memory_utils function only takes 3 parameters but we need the callback functionality,
            # perform the memory check ourselves after calling the external function
            if memory_threshold is not None and trim_callback is not None:
                try:
                    memory_usage = psutil.virtual_memory().percent
                    if memory_usage > memory_threshold:
                        trim_callback(memory_usage)
                except Exception as e:
                    logger.warning(
                        f"Memory check callback error for {component_id}: {e}"
                    )

            return result
    except ImportError:
        # Fall back to basic implementation
        logger.debug(
            f"memory_utils not available, using basic memory check for {component_id}"
        )

        # Simple check without trying to coordinate with other components
        if memory_threshold is not None and trim_callback is not None:
            try:
                # Use a pseudorandom approach based on component_id to avoid check storms
                # This distributes memory checks across time rather than having all components
                # check simultaneously, even without a central coordinator
                check_hash = hash(component_id) % 100
                time_hash = int(current_time) % 60  # Distribute checks over a minute

                # Only check if the component's hash matches the current time pattern
                if check_hash <= (
                    time_hash * 1.7
                ):  # Similar to MemoryManager's distribution algorithm
                    memory_usage = psutil.virtual_memory().percent
                    if memory_usage > memory_threshold:
                        trim_callback(memory_usage)
            except Exception as e:
                logger.warning(f"Memory check error for {component_id}: {e}")

        return current_time  # Update last check time


def optimize_gpu_memory() -> bool:
    """
    Optimize GPU memory usage by clearing unused tensors and cache.

    Returns:
        bool: True if optimization was successful
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Get memory stats before and after
        before = torch.cuda.memory_allocated()
        gc.collect()  # Run CPU garbage collection which can help release GPU memory
        torch.cuda.empty_cache()  # Clear cache again after GC
        after = torch.cuda.memory_allocated()

        savings = (before - after) / 1024**2  # Convert to MB
        if savings > 1:  # Only log if we saved at least 1MB
            logger.info(f"GPU memory optimization freed {savings:.1f} MB")

        return True
    except Exception as e:
        logger.warning(f"GPU memory optimization failed: {e}")
        return False


def adaptive_batch_size_calculator(matrix1: np.ndarray, matrix2: np.ndarray) -> int:
    """
    Calculate an optimal batch size for matrix operations based on matrix dimensions.

    Args:
        matrix1: First input matrix
        matrix2: Second input matrix

    Returns:
        int: Recommended batch size
    """
    # Estimate memory usage based on matrix dimensions
    row_count = matrix1.shape[0]
    col_count = matrix2.shape[0]
    dimension = matrix1.shape[1]

    # Calculate total elements in matrices and result
    total_elements = (
        row_count * dimension + col_count * dimension + row_count * col_count
    )

    # Heuristic: larger matrices need smaller batch sizes to avoid memory issues
    if total_elements > 10000000:  # Very large
        batch_size = 50
    elif total_elements > 1000000:  # Large
        batch_size = 100
    elif total_elements > 100000:  # Medium
        batch_size = 200
    else:  # Small
        batch_size = 500

    # Scale based on row count and ensure reasonable bounds
    return max(10, min(batch_size, row_count // 4 + 1))


class CacheKey:
    """
    Utility for creating standardized cache keys with built-in versioning and namespacing.

    Ensures consistent key generation and allows for key inspection.
    """

    @staticmethod
    def create(
        namespace: str, key: str, config_hash: str = "", version: str = CACHE_VERSION
    ) -> str:
        """Create a standardized cache key."""
        config_part = f":{config_hash[:8]}" if config_hash else ""
        return f"{version}:{namespace}{config_part}:{key}"

    @staticmethod
    def parse(full_key: str) -> CacheKeyInfo:
        """Parse a standardized cache key into its components."""
        try:
            parts = full_key.split(":", 3)
            if len(parts) < 3:
                return {"valid": False}

            version, namespace, key = parts[0], parts[1], parts[-1]
            config_hash = ""

            # Extract config hash if present
            if len(parts) == 4:
                config_hash = parts[2]

            return {
                "valid": True,
                "version": version,
                "namespace": namespace,
                "config_hash": config_hash,
                "key": key,
            }
        except Exception:
            return {"valid": False}


# -----------------------------------------------------------------------------
# Serialization Manager
# -----------------------------------------------------------------------------


class SerializationManager:
    """
    Manages serialization/deserialization for cache entries based on data type.
    Optimizes disk storage while maintaining raw objects in memory for performance.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration options."""
        self.config = config or {}

        # Check available serialization libraries
        self.use_msgpack = self._check_import("srsly")
        self.use_flexbuffers = self._check_import("flexbuffers")

        # Configuration options with defaults
        self.use_numpy_optimized = self.config.get("use_numpy_optimized", True)
        self.prefer_flexbuffers_for_vectors = self.config.get(
            "prefer_flexbuffers_for_vectors", False
        )

        # Security integration
        self.use_encryption = self.config.get(
            "use_encryption", SECURITY_MODULE_AVAILABLE
        )
        self.secure_serializer = None
        if self.use_encryption and SECURITY_MODULE_AVAILABLE:
            try:
                self.secure_serializer = SecureCacheSerializer()
                logger.debug("Secure serialization enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize secure serializer: {e}")
                self.use_encryption = False

        logger.debug(
            f"SerializationManager initialized with msgpack={self.use_msgpack}, "
            f"flexbuffers={self.use_flexbuffers}, "
            f"encryption={self.use_encryption}"
        )

    def _check_import(self, module_name: str) -> bool:
        """Check if a module is available."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def serialize(self, value: Any) -> Tuple[bytes, str]:
        """
        Serialize a value using the most appropriate method based on data type.

        Returns:
            Tuple[bytes, str]: Serialized data and format identifier
        """
        # Handle NumPy arrays
        if isinstance(value, np.ndarray):
            data, format_id = self._serialize_numpy_array(value)

        # Handle PyTorch tensors
        elif torch.is_tensor(value):
            data, format_id = self._serialize_torch_tensor(value)

        # Use msgpack for general data if available
        elif self.use_msgpack:
            try:
                import srsly

                data = srsly.msgpack_dumps(value)
                format_id = "msgpack"
            except Exception as e:
                logger.debug(
                    f"MessagePack serialization failed: {e}, falling back to JSON"
                )
                data = json.dumps(value).encode("utf-8")
                format_id = "json"

        # Fall back to JSON for general compatibility
        else:
            data = json.dumps(value).encode("utf-8")
            format_id = "json"

        # Apply encryption if enabled
        if self.use_encryption and self.secure_serializer:
            try:
                encrypted_data = self.secure_serializer.encrypt(data)
                return encrypted_data, f"encrypted_{format_id}"
            except Exception as e:
                logger.warning(f"Encryption failed: {e}, using unencrypted data")

        return data, format_id

    def deserialize(self, data: bytes, format_id: str) -> Any:
        """
        Deserialize data using the specified format.
        """
        if not data:
            return None

        # Handle encryption
        if format_id.startswith("encrypted_") and self.secure_serializer:
            try:
                data = self.secure_serializer.decrypt(data)
                format_id = format_id[10:]  # Remove "encrypted_" prefix
            except Exception as e:
                logger.error(f"Decryption error: {e}")
                raise ValueError(f"Failed to decrypt data: {e}")

        # Handle different formats
        if format_id == "msgpack":
            import srsly

            return srsly.msgpack_loads(data)

        elif format_id == "msgpack_numpy":
            if self.use_msgpack:
                import srsly

                return srsly.msgpack_loads(data)
            else:
                return self._fallback_deserialize_numpy(data)

        elif format_id == "flexbuffers":
            if self.use_flexbuffers:
                import flexbuffers

                return flexbuffers.loads(data)
            else:
                raise ValueError(
                    "FlexBuffers deserialization requested but not available"
                )

        elif format_id == "json":
            return json.loads(data.decode("utf-8"))

        elif format_id == "numpy_raw":
            # Extract dtype and shape information from header
            dtype_info_size = int.from_bytes(data[:4], byteorder="little")
            dtype_str = data[4 : 4 + dtype_info_size].decode("utf-8")
            shape_info_size = int.from_bytes(
                data[4 + dtype_info_size : 8 + dtype_info_size], byteorder="little"
            )
            shape_str = data[
                8 + dtype_info_size : 8 + dtype_info_size + shape_info_size
            ].decode("utf-8")

            # Parse dtype and shape
            dtype = np.dtype(dtype_str)
            shape = tuple(map(int, shape_str.split(",")))

            # Extract array data
            array_data = data[8 + dtype_info_size + shape_info_size :]

            # Reconstruct array
            return np.frombuffer(array_data, dtype=dtype).reshape(shape)

        else:
            raise ValueError(f"Unknown serialization format: {format_id}")

    def _serialize_numpy_array(self, array: np.ndarray) -> Tuple[bytes, str]:
        """Serialize a numpy array with the most efficient method available."""
        # Use msgpack-numpy through srsly if available
        if self.use_msgpack and self.use_numpy_optimized:
            try:
                import srsly

                serialized = srsly.msgpack_dumps(array)
                return serialized, "msgpack_numpy"
            except Exception as e:
                logger.debug(
                    f"msgpack-numpy serialization failed: {e}, using alternatives"
                )

        # Try flexbuffers for numeric arrays when configured
        if self.use_flexbuffers and self.prefer_flexbuffers_for_vectors:
            try:
                import flexbuffers

                serialized = flexbuffers.dumps(array.tolist())
                return serialized, "flexbuffers"
            except Exception as e:
                logger.debug(f"FlexBuffer serialization failed: {e}, using fallback")

        # Raw numpy serialization - compact and efficient
        try:
            # Store dtype as string
            dtype_str = str(array.dtype).encode("utf-8")
            dtype_size = len(dtype_str)
            dtype_header = dtype_size.to_bytes(4, byteorder="little")

            # Store shape
            shape_str = ",".join(map(str, array.shape)).encode("utf-8")
            shape_size = len(shape_str)
            shape_header = shape_size.to_bytes(4, byteorder="little")

            # Get binary data
            array_bytes = array.tobytes()

            # Combine into single byte sequence
            serialized = (
                dtype_header + dtype_str + shape_header + shape_str + array_bytes
            )
            return serialized, "numpy_raw"
        except Exception as e:
            logger.warning(f"Raw numpy serialization failed: {e}, falling back to JSON")
            # Last resort - convert to list and use JSON
            serialized = json.dumps(array.tolist()).encode("utf-8")
            return serialized, "json"

    def _serialize_torch_tensor(self, tensor: torch.Tensor) -> Tuple[bytes, str]:
        """Serialize a torch tensor by converting to numpy first."""
        try:
            # Handle gradient and device placement
            if tensor.requires_grad:
                tensor = tensor.detach()
            if tensor.is_cuda:
                tensor = tensor.cpu()

            # Convert to numpy and use numpy serialization
            return self._serialize_numpy_array(tensor.numpy())
        except Exception as e:
            logger.warning(f"Torch tensor serialization error: {e}")
            # Fallback - convert to list
            serialized = json.dumps(tensor.tolist()).encode("utf-8")
            return serialized, "json"

    def _fallback_deserialize_numpy(self, data: bytes) -> np.ndarray:
        """
        Fallback method for numpy array deserialization when srsly is unavailable.
        Handles both raw binary format and JSON format for better robustness.
        """
        try:
            # First attempt to parse as raw NumPy format
            try:
                # Extract dtype and shape information from header
                dtype_info_size = int.from_bytes(data[:4], byteorder="little")
                dtype_str = data[4 : 4 + dtype_info_size].decode("utf-8")
                shape_info_size = int.from_bytes(
                    data[4 + dtype_info_size : 8 + dtype_info_size], byteorder="little"
                )
                shape_str = data[
                    8 + dtype_info_size : 8 + dtype_info_size + shape_info_size
                ].decode("utf-8")

                # Parse dtype and shape
                dtype = np.dtype(dtype_str)
                shape = tuple(map(int, shape_str.split(",")))

                # Extract array data
                array_data = data[8 + dtype_info_size + shape_info_size :]

                # Reconstruct array
                return np.frombuffer(array_data, dtype=dtype).reshape(shape)
            except (ValueError, IndexError, TypeError, UnicodeDecodeError):
                # If raw format parsing fails, try JSON format
                array_list = json.loads(data.decode("utf-8"))
                return np.array(array_list)
        except Exception as e:
            logger.error(f"Failed to deserialize numpy array: {e}", exc_info=True)
            raise ValueError(f"Could not deserialize numpy array: {e}")


# -----------------------------------------------------------------------------
# Cache Backends
# -----------------------------------------------------------------------------


class CacheBackend:
    """Abstract base class for cache backends."""

    def get(self, key: str) -> Optional[Any]:
        """Retrieve an item from the cache."""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store an item in the cache with optional TTL."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all items from the cache."""
        raise NotImplementedError

    def contains(self, key: str) -> bool:
        """Check if key exists in cache without retrieving value."""
        raise NotImplementedError

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        raise NotImplementedError

    def trim(self, percent: float = 10.0) -> int:
        """Trim percentage of cache entries to free memory."""
        raise NotImplementedError

    def adjust_capacity(self, new_size: int) -> None:
        """Adjust the cache capacity."""
        raise NotImplementedError

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple items from the cache at once."""
        raise NotImplementedError

    def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store multiple items in the cache at once."""
        raise NotImplementedError


class MemoryCacheBackend(CacheBackend):
    """Memory-only cache backend with metrics and memory monitoring."""

    def __init__(self, config: Dict[str, Any], namespace: str = "general"):
        # Extract config with fallbacks
        self.namespace = namespace
        self.config = config.get("caching", {})
        component_config = self.config.get(namespace, {})

        # Set up memory cache
        self.memory_size = component_config.get(
            "memory_size", self.config.get("cache_size", 1000)
        )
        self.ttl = component_config.get("ttl", 3600)

        # Use TTLCache if TTL is specified, otherwise LRUCache
        if self.ttl > 0:
            self.memory_cache = TTLCache(maxsize=self.memory_size, ttl=self.ttl)
        else:
            self.memory_cache = LRUCache(maxsize=self.memory_size)

        # Add memory monitoring
        self.last_memory_check = time.time()
        self.memory_check_interval = config.get("caching", {}).get(
            "memory_check_interval", 300
        )  # 5 minutes
        self.max_memory_percent = config.get("hardware_limits", {}).get(
            "max_ram_usage_percent", 80
        )

        # Add lock for thread safety with proper registration if lock_utils is available
        if LOCK_UTILS_AVAILABLE:
            self._lock = create_component_lock(f"memory_cache_{namespace}")
        else:
            self._lock = RLock()

        # Metrics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
        }

    def get(self, key: str) -> Any:
        """Get item from cache with metrics tracking."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"memory_cache_{self.namespace}_get"
            ):
                # Try memory cache
                if key in self.memory_cache:
                    self.stats["hits"] += 1
                    return self.memory_cache[key]

                self.stats["misses"] += 1

                # Periodically check memory usage
                self._check_memory_usage()

                return None
        else:
            # Original implementation with regular lock
            with self._lock:
                # Try memory cache
                if key in self.memory_cache:
                    self.stats["hits"] += 1
                    return self.memory_cache[key]

                self.stats["misses"] += 1

                # Periodically check memory usage
                self._check_memory_usage()

                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache with correct TTL handling."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"memory_cache_{self.namespace}_set"
            ):
                # Check memory usage before adding new items
                self._check_memory_usage()

                # Store in memory cache with proper TTL handling
                if ttl is not None and isinstance(self.memory_cache, TTLCache):
                    # Store the value
                    self.memory_cache[key] = value

                    # Override the TTL if different from default and if timer is accessible
                    if ttl != self.ttl and hasattr(self.memory_cache, "timer"):
                        current_time = time.time()
                        self.memory_cache.timer[key] = current_time + ttl
                else:
                    self.memory_cache[key] = value

                self.stats["size"] = len(self.memory_cache)
                return key in self.memory_cache
        else:
            # Original implementation with regular lock
            with self._lock:
                # Check memory usage before adding new items
                self._check_memory_usage()

                # Store in memory cache with proper TTL handling
                if ttl is not None and isinstance(self.memory_cache, TTLCache):
                    # Store the value
                    self.memory_cache[key] = value

                    # Override the TTL if different from default and if timer is accessible
                    if ttl != self.ttl and hasattr(self.memory_cache, "timer"):
                        current_time = time.time()
                        self.memory_cache.timer[key] = current_time + ttl
                else:
                    self.memory_cache[key] = value

                self.stats["size"] = len(self.memory_cache)
                return key in self.memory_cache

    def clear(self) -> None:
        """Clear the cache."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"memory_cache_{self.namespace}_clear"
            ):
                self.memory_cache.clear()
                self.stats["size"] = 0
                self.stats["evictions"] += 1  # Count clearing as an eviction event
        else:
            with self._lock:
                self.memory_cache.clear()
                self.stats["size"] = 0
                self.stats["evictions"] += 1  # Count clearing as an eviction event

    def contains(self, key: str) -> bool:
        """Check if key exists in cache without retrieval."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"memory_cache_{self.namespace}_contains"
            ):
                return key in self.memory_cache
        else:
            with self._lock:
                return key in self.memory_cache

    def get_stats(self) -> "CacheStats":
        """Get cache statistics."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"memory_cache_{self.namespace}_stats"
            ):
                stats_copy = self.stats.copy()
                total_ops = stats_copy["hits"] + stats_copy["misses"]

                if total_ops > 0:
                    stats_copy["hit_ratio"] = stats_copy["hits"] / total_ops
                else:
                    stats_copy["hit_ratio"] = 0

                stats_copy["capacity"] = self.memory_size
                stats_copy["usage_percent"] = (
                    (len(self.memory_cache) / self.memory_size * 100)
                    if self.memory_size > 0
                    else 0
                )

                return cast("CacheStats", stats_copy)
        else:
            with self._lock:
                stats_copy = self.stats.copy()
                total_ops = stats_copy["hits"] + stats_copy["misses"]

                if total_ops > 0:
                    stats_copy["hit_ratio"] = stats_copy["hits"] / total_ops
                else:
                    stats_copy["hit_ratio"] = 0

                stats_copy["capacity"] = self.memory_size
                stats_copy["usage_percent"] = (
                    (len(self.memory_cache) / self.memory_size * 100)
                    if self.memory_size > 0
                    else 0
                )

                return cast("CacheStats", stats_copy)

    def trim(self, percent: float = 10.0) -> int:
        """Trim percentage of cache entries to free memory."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"memory_cache_{self.namespace}_trim"
            ):
                if not self.memory_cache:
                    return 0

                trim_count = int(len(self.memory_cache) * percent / 100)
                if trim_count <= 0:
                    return 0

                for _ in range(trim_count):
                    try:
                        self.memory_cache.popitem()
                        self.stats["evictions"] += 1
                    except KeyError:
                        break  # Cache is empty

                self.stats["size"] = len(self.memory_cache)
                return trim_count
        else:
            with self._lock:
                if not self.memory_cache:
                    return 0

                trim_count = int(len(self.memory_cache) * percent / 100)
                if trim_count <= 0:
                    return 0

                for _ in range(trim_count):
                    try:
                        self.memory_cache.popitem()
                        self.stats["evictions"] += 1
                    except KeyError:
                        break  # Cache is empty

                self.stats["size"] = len(self.memory_cache)
                return trim_count

    def adjust_capacity(self, new_size: int) -> None:
        """Adjust the cache capacity."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"memory_cache_{self.namespace}_adjust_capacity"
            ):
                if new_size == self.memory_size:
                    return  # No change needed

                # Create new cache with adjusted size
                old_cache = self.memory_cache
                self.memory_size = max(1, new_size)  # Ensure positive size

                # Create appropriate cache type
                if self.ttl > 0:
                    self.memory_cache = TTLCache(maxsize=self.memory_size, ttl=self.ttl)
                else:
                    self.memory_cache = LRUCache(maxsize=self.memory_size)

                # Copy over items if needed
                if new_size > 0:
                    # Get sorted items to prioritize
                    items_to_copy = list(old_cache.items())

                    # For TTL cache, sort by expiration time
                    if hasattr(old_cache, "timer") and callable(
                        getattr(old_cache, "timer", None)
                    ):
                        # Sort by time remaining - keep items with most time left
                        current_time = time.time()
                        try:
                            items_to_copy.sort(
                                key=lambda x: old_cache.timer.get(x[0], 0)
                                - current_time,
                                reverse=True,
                            )
                        except (AttributeError, TypeError):
                            pass  # Fall back to unsorted

                    # Copy up to new capacity
                    for key, value in items_to_copy[:new_size]:
                        self.memory_cache[key] = value

                self.stats["size"] = len(self.memory_cache)

                logger.info(
                    f"Cache {self.namespace} resized from {len(old_cache)} to {self.memory_size} items"
                )
        else:
            with self._lock:
                if new_size == self.memory_size:
                    return  # No change needed

                # Create new cache with adjusted size
                old_cache = self.memory_cache
                self.memory_size = max(1, new_size)  # Ensure positive size

                # Create appropriate cache type
                if self.ttl > 0:
                    self.memory_cache = TTLCache(maxsize=self.memory_size, ttl=self.ttl)
                else:
                    self.memory_cache = LRUCache(maxsize=self.memory_size)

                # Copy over items if needed
                if new_size > 0:
                    # Get sorted items to prioritize
                    items_to_copy = list(old_cache.items())

                    # For TTL cache, sort by expiration time
                    if hasattr(old_cache, "timer") and callable(
                        getattr(old_cache, "timer", None)
                    ):
                        # Sort by time remaining - keep items with most time left
                        current_time = time.time()
                        try:
                            items_to_copy.sort(
                                key=lambda x: old_cache.timer.get(x[0], 0)
                                - current_time,
                                reverse=True,
                            )
                        except (AttributeError, TypeError):
                            pass  # Fall back to unsorted

                    # Copy up to new capacity
                    for key, value in items_to_copy[:new_size]:
                        self.memory_cache[key] = value

                self.stats["size"] = len(self.memory_cache)

                logger.info(
                    f"Cache {self.namespace} resized from {len(old_cache)} to {self.memory_size} items"
                )

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple items from the cache at once."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"memory_cache_{self.namespace}_get_many"
            ):
                result = {}
                for key in keys:
                    if key in self.memory_cache:
                        result[key] = self.memory_cache[key]
                        self.stats["hits"] += 1
                    else:
                        self.stats["misses"] += 1

                # Periodically check memory usage after batch operations
                if len(keys) > 10:  # Only check after significant operations
                    self._check_memory_usage()

                return result
        else:
            with self._lock:
                result = {}
                for key in keys:
                    if key in self.memory_cache:
                        result[key] = self.memory_cache[key]
                        self.stats["hits"] += 1
                    else:
                        self.stats["misses"] += 1

                # Periodically check memory usage after batch operations
                if len(keys) > 10:  # Only check after significant operations
                    self._check_memory_usage()

                return result

    def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store multiple items in the cache at once."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"memory_cache_{self.namespace}_set_many"
            ):
                success = True
                # Check memory usage before batch operation
                self._check_memory_usage()

                for key, value in items.items():
                    if not self.set(key, value, ttl):
                        success = False

                return success
        else:
            with self._lock:
                success = True
                # Check memory usage before batch operation
                self._check_memory_usage()

                for key, value in items.items():
                    if not self.set(key, value, ttl):
                        success = False

                return success

    def _check_memory_usage(self) -> None:
        """Monitor and optimize memory usage using the centralized function."""
        component_id = f"cache_{self.namespace}_{id(self)}"
        updated_time = check_memory_usage(
            component_id, self.last_memory_check, self.memory_check_interval
        )

        # Only proceed with check if enough time has passed
        if updated_time > self.last_memory_check:
            self.last_memory_check = updated_time

            try:
                memory_usage = psutil.virtual_memory().percent

                # Check if we need to trim the cache
                if memory_usage > self.max_memory_percent:
                    # Keep only 75% of the items in memory cache
                    if len(self.memory_cache) > 100:
                        try:
                            trim_count = int(
                                len(self.memory_cache) * 0.25
                            )  # Remove 25%
                            self.trim(25.0)
                            logger.info(
                                f"Trimmed {trim_count} items from {self.namespace} cache"
                            )
                        except Exception as e:
                            logger.error(f"Error during cache trimming: {e}")
            except Exception as e:
                logger.warning(f"Error checking memory usage: {e}")


class HybridCacheBackend(CacheBackend):
    """Cache backend that combines memory and disk caching with metrics."""

    def __init__(self, config: Dict[str, Any], namespace: str = "general"):
        # Extract config with fallbacks
        self.namespace = namespace
        self.config = config.get("caching", {})
        component_config = self.config.get(namespace, {})

        # Set up memory cache
        self.memory_size = component_config.get(
            "memory_size", self.config.get("cache_size", 1000)
        )
        self.ttl = component_config.get("ttl", 3600)

        # Use TTLCache if TTL is specified, otherwise LRUCache
        if self.ttl > 0:
            self.memory_cache = TTLCache(maxsize=self.memory_size, ttl=self.ttl)
        else:
            self.memory_cache = LRUCache(maxsize=self.memory_size)

        # Initialize serialization manager
        self.serializer = SerializationManager(config.get("serialization", {}))

        # Set up disk cache if configured
        self.cache_dir = self.config.get("cache_dir")
        self.disk_required = self.config.get("require_disk_cache", True)
        self.disk_cache = None

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            try:
                # Configure disk cache to minimize serialization overhead
                self.disk_cache = Cache(
                    os.path.join(self.cache_dir, namespace),
                    disk_pickle_protocol=4,  # Recent protocol version
                    disk_min_file_size=1,  # Minimize overhead
                    statistics=True,  # Keep statistics
                )
                # Check if disk cache is writable
                self.disk_cache["_test_key"] = 1
                del self.disk_cache["_test_key"]
                logger.debug(
                    f"Disk cache initialized successfully at {os.path.join(self.cache_dir, namespace)}"
                )
            except (OSError, IOError, PermissionError) as e:
                error_msg = f"Could not initialize disk cache: {e}"
                if self.disk_required:
                    logger.error(
                        f"{error_msg} - Disk cache is required, raising exception."
                    )
                    raise RuntimeError(f"Disk cache initialization failed: {e}")
                else:
                    logger.warning(f"{error_msg} - Falling back to memory-only cache.")
                    self.disk_cache = None

        # Circuit breaker for disk operations
        self.disk_failures = 0
        self.disk_failure_window = []  # Track recent failures with timestamps
        self.max_failures = self.config.get("max_retries", 3)
        self.disk_enabled = self.disk_cache is not None

        # Log operating mode
        if self.disk_enabled:
            logger.debug(
                f"HybridCacheBackend '{namespace}' operating in hybrid (memory+disk) mode"
            )
        else:
            logger.info(
                f"HybridCacheBackend '{namespace}' operating in memory-only mode"
            )

        # Metrics
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "disk_errors": 0,
            "evictions": 0,
            "size": 0,
            "latency_sum": 0.0,
            "latency_count": 0,
            "max_latency": 0.0,
        }

        # Add memory monitoring
        self.last_memory_check = time.time()
        self.memory_check_interval = config.get("caching", {}).get(
            "memory_check_interval", 300
        )  # 5 minutes
        self.max_memory_percent = config.get("hardware_limits", {}).get(
            "max_ram_usage_percent", 80
        )

        # Add lock for thread safety with proper registration if lock_utils is available
        if LOCK_UTILS_AVAILABLE:
            self._lock = create_component_lock(f"hybrid_cache_{namespace}")
        else:
            self._lock = RLock()

    def get(self, key: str) -> Any:
        """Get item from cache with optimized deserialization."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_get"
            ):
                # Try memory cache first for raw objects
                if key in self.memory_cache:
                    self.stats["memory_hits"] += 1
                    return self.memory_cache[key]

                self.stats["memory_misses"] += 1

                # Try disk cache if available and enabled
                if self.disk_cache and self.disk_enabled:
                    try:
                        with self._track_latency():
                            if key in self.disk_cache:
                                # Get serialized value
                                serialized_value = self.disk_cache[key]

                                # Check if it's a serialization tuple (data, format_id)
                                if (
                                    isinstance(serialized_value, tuple)
                                    and len(serialized_value) == 2
                                ):
                                    data, format_id = serialized_value
                                    # Deserialize and store raw value in memory
                                    try:
                                        value = self.serializer.deserialize(
                                            data, format_id
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"Deserialization failed for {key}: {e}"
                                        )
                                        self.stats["disk_errors"] += 1
                                        return None
                                else:
                                    # Legacy data format
                                    value = serialized_value

                                # Promote to memory cache
                                self.memory_cache[key] = value
                                self.stats["disk_hits"] += 1
                                # Mark successful disk operation
                                self._handle_disk_success()
                                return value
                        self.stats["disk_misses"] += 1
                    except Exception as e:
                        self._handle_disk_error(f"Disk cache read error: {e}")

                # Periodically check memory usage and evict items if needed
                self._check_memory_usage()

                return None
        else:
            with self._lock:
                # Try memory cache first
                if key in self.memory_cache:
                    self.stats["memory_hits"] += 1
                    return self.memory_cache[key]

                self.stats["memory_misses"] += 1

                # Try disk cache if available and enabled
                if self.disk_cache and self.disk_enabled:
                    try:
                        with self._track_latency():
                            if key in self.disk_cache:
                                # Get serialized value
                                serialized_value = self.disk_cache[key]

                                # Check if it's a serialization tuple (data, format_id)
                                if (
                                    isinstance(serialized_value, tuple)
                                    and len(serialized_value) == 2
                                ):
                                    data, format_id = serialized_value
                                    # Deserialize and store raw value in memory
                                    try:
                                        value = self.serializer.deserialize(
                                            data, format_id
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"Deserialization failed for {key}: {e}"
                                        )
                                        self.stats["disk_errors"] += 1
                                        return None
                                else:
                                    # Legacy data format
                                    value = serialized_value

                                # Promote to memory cache
                                self.memory_cache[key] = value
                                self.stats["disk_hits"] += 1
                                # Mark successful disk operation
                                self._handle_disk_success()
                                return value
                        self.stats["disk_misses"] += 1
                    except Exception as e:
                        self._handle_disk_error(f"Disk cache read error: {e}")

                # Periodically check memory usage and evict items if needed
                self._check_memory_usage()

                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache with efficient serialization for disk."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_set"
            ):
                # Always store raw value in memory for performance
                self.memory_cache[key] = value
                self.stats["size"] = len(self.memory_cache)

                # Store serialized value in disk if available and enabled
                if self.disk_cache and self.disk_enabled:
                    try:
                        with self._track_latency():
                            # Serialize the value for efficient disk storage
                            data, format_id = self.serializer.serialize(value)
                            self.disk_cache[key] = (data, format_id)

                            if ttl is not None:
                                self.disk_cache.touch(key, ttl)
                        # Mark successful disk operation
                        self._handle_disk_success()
                        return True
                    except Exception as e:
                        self._handle_disk_error(f"Disk cache write error: {e}")

                # Check memory usage before adding new items
                self._check_memory_usage()

                return key in self.memory_cache
        else:
            with self._lock:
                # Always store raw value in memory for performance
                self.memory_cache[key] = value
                self.stats["size"] = len(self.memory_cache)

                # Store serialized value in disk if available and enabled
                if self.disk_cache and self.disk_enabled:
                    try:
                        with self._track_latency():
                            # Serialize the value for efficient disk storage
                            data, format_id = self.serializer.serialize(value)
                            self.disk_cache[key] = (data, format_id)

                            if ttl is not None:
                                self.disk_cache.touch(key, ttl)
                        # Mark successful disk operation
                        self._handle_disk_success()
                        return True
                    except Exception as e:
                        self._handle_disk_error(f"Disk cache write error: {e}")

                # Check memory usage before adding new items
                self._check_memory_usage()

                return key in self.memory_cache

    def clear(self) -> None:
        """Clear all caches."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_clear"
            ):
                self.memory_cache.clear()
                self.stats["size"] = 0
                self.stats["evictions"] += 1

                if self.disk_cache and self.disk_enabled:
                    try:
                        self.disk_cache.clear()
                    except Exception as e:
                        logger.error(f"Error clearing disk cache: {e}")
                        self._handle_disk_error(f"Disk cache clear error: {e}")
        else:
            with self._lock:
                self.memory_cache.clear()
                self.stats["size"] = 0
                self.stats["evictions"] += 1

                if self.disk_cache and self.disk_enabled:
                    try:
                        self.disk_cache.clear()
                    except Exception as e:
                        logger.error(f"Error clearing disk cache: {e}")
                        self._handle_disk_error(f"Disk cache clear error: {e}")

    def contains(self, key: str) -> bool:
        """Check if key exists in cache without retrieving the value."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_contains"
            ):
                if key in self.memory_cache:
                    return True

                if self.disk_cache and self.disk_enabled:
                    try:
                        return key in self.disk_cache
                    except Exception as e:
                        self._handle_disk_error(f"Disk cache check error: {e}")

                return False
        else:
            with self._lock:
                if key in self.memory_cache:
                    return True

                if self.disk_cache and self.disk_enabled:
                    try:
                        return key in self.disk_cache
                    except Exception as e:
                        self._handle_disk_error(f"Disk cache check error: {e}")

                return False

    def get_stats(self) -> "CacheStats":
        """Get cache statistics."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_stats"
            ):
                stats_copy = self.stats.copy()

                # Calculate hit ratios
                mem_total = stats_copy["memory_hits"] + stats_copy["memory_misses"]
                disk_total = stats_copy["disk_hits"] + stats_copy["disk_misses"]

                stats_copy["memory_hit_ratio"] = (
                    stats_copy["memory_hits"] / mem_total if mem_total > 0 else 0
                )
                stats_copy["disk_hit_ratio"] = (
                    stats_copy["disk_hits"] / disk_total if disk_total > 0 else 0
                )
                stats_copy["overall_hit_ratio"] = (
                    (stats_copy["memory_hits"] + stats_copy["disk_hits"])
                    / (mem_total + disk_total)
                    if (mem_total + disk_total) > 0
                    else 0
                )

                # Add average latency metric
                if stats_copy["latency_count"] > 0:
                    stats_copy["avg_latency"] = (
                        stats_copy["latency_sum"] / stats_copy["latency_count"]
                    )
                else:
                    stats_copy["avg_latency"] = 0.0

                # Add capacity information
                stats_copy["memory_capacity"] = self.memory_size
                stats_copy["memory_usage_percent"] = (
                    (len(self.memory_cache) / self.memory_size * 100)
                    if self.memory_size > 0
                    else 0
                )

                # Add disk stats if available
                if self.disk_cache and self.disk_enabled:
                    try:
                        stats_copy["disk_size"] = len(self.disk_cache)
                        stats_copy["disk_available"] = self.disk_enabled
                    except Exception:
                        stats_copy["disk_available"] = False
                else:
                    stats_copy["disk_available"] = False

                return cast("CacheStats", stats_copy)
        else:
            with self._lock:
                stats_copy = self.stats.copy()

                # Calculate hit ratios
                mem_total = stats_copy["memory_hits"] + stats_copy["memory_misses"]
                disk_total = stats_copy["disk_hits"] + stats_copy["disk_misses"]

                stats_copy["memory_hit_ratio"] = (
                    stats_copy["memory_hits"] / mem_total if mem_total > 0 else 0
                )
                stats_copy["disk_hit_ratio"] = (
                    stats_copy["disk_hits"] / disk_total if disk_total > 0 else 0
                )
                stats_copy["overall_hit_ratio"] = (
                    (stats_copy["memory_hits"] + stats_copy["disk_hits"])
                    / (mem_total + disk_total)
                    if (mem_total + disk_total) > 0
                    else 0
                )

                # Add average latency metric
                if stats_copy["latency_count"] > 0:
                    stats_copy["avg_latency"] = (
                        stats_copy["latency_sum"] / stats_copy["latency_count"]
                    )
                else:
                    stats_copy["avg_latency"] = 0.0

                # Add capacity information
                stats_copy["memory_capacity"] = self.memory_size
                stats_copy["memory_usage_percent"] = (
                    (len(self.memory_cache) / self.memory_size * 100)
                    if self.memory_size > 0
                    else 0
                )

                # Add disk stats if available
                if self.disk_cache and self.disk_enabled:
                    try:
                        stats_copy["disk_size"] = len(self.disk_cache)
                        stats_copy["disk_available"] = self.disk_enabled
                    except Exception:
                        stats_copy["disk_available"] = False
                else:
                    stats_copy["disk_available"] = False

                return cast("CacheStats", stats_copy)

    def trim(self, percent: float = 10.0) -> int:
        """Trim percentage of cache entries to free memory."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_trim"
            ):
                # Calculate how many items to remove
                if not self.memory_cache:
                    return 0

                trim_count = int(len(self.memory_cache) * percent / 100)
                if trim_count <= 0:
                    return 0

                # Remove items from memory cache
                removed = 0
                for _ in range(trim_count):
                    try:
                        self.memory_cache.popitem()
                        removed += 1
                        self.stats["evictions"] += 1
                    except KeyError:
                        break  # Cache is empty

                self.stats["size"] = len(self.memory_cache)
                return removed
        else:
            with self._lock:
                # Calculate how many items to remove
                if not self.memory_cache:
                    return 0

                trim_count = int(len(self.memory_cache) * percent / 100)
                if trim_count <= 0:
                    return 0

                # Remove items from memory cache
                removed = 0
                for _ in range(trim_count):
                    try:
                        self.memory_cache.popitem()
                        removed += 1
                        self.stats["evictions"] += 1
                    except KeyError:
                        break  # Cache is empty

                self.stats["size"] = len(self.memory_cache)
                return removed

    def adjust_capacity(self, new_size: int) -> None:
        """Adjust the memory cache capacity."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_adjust_capacity"
            ):
                if new_size == self.memory_size:
                    return  # No change needed

                # Create new cache with adjusted size
                old_cache = self.memory_cache
                self.memory_size = max(1, new_size)  # Ensure positive size

                # Create appropriate cache type
                if self.ttl > 0:
                    self.memory_cache = TTLCache(maxsize=self.memory_size, ttl=self.ttl)
                else:
                    self.memory_cache = LRUCache(maxsize=self.memory_size)

                # Copy over items if needed
                if new_size > 0:
                    # Sort by recency if possible
                    items_to_copy = list(old_cache.items())

                    # For TTL cache, sort by expiration time
                    if hasattr(old_cache, "timer") and callable(
                        getattr(old_cache, "timer", None)
                    ):
                        # Sort by time remaining - keep items with most time left
                        current_time = time.time()
                        try:
                            items_to_copy.sort(
                                key=lambda x: old_cache.timer.get(x[0], 0)
                                - current_time,
                                reverse=True,
                            )
                        except (AttributeError, TypeError):
                            pass  # Fall back to unsorted

                    # Transfer most recent items first, up to new capacity
                    for key, value in items_to_copy[:new_size]:
                        self.memory_cache[key] = value

                self.stats["size"] = len(self.memory_cache)
                logger.info(
                    f"Cache {self.namespace} resized from {len(old_cache)} to {new_size} items"
                )
        else:
            with self._lock:
                if new_size == self.memory_size:
                    return  # No change needed

                # Create new cache with adjusted size
                old_cache = self.memory_cache
                self.memory_size = max(1, new_size)  # Ensure positive size

                # Create appropriate cache type
                if self.ttl > 0:
                    self.memory_cache = TTLCache(maxsize=self.memory_size, ttl=self.ttl)
                else:
                    self.memory_cache = LRUCache(maxsize=self.memory_size)

                # Copy over items if needed
                if new_size > 0:
                    # Sort by recency if possible
                    items_to_copy = list(old_cache.items())

                    # For TTL cache, sort by expiration time
                    if hasattr(old_cache, "timer") and callable(
                        getattr(old_cache, "timer", None)
                    ):
                        # Sort by time remaining - keep items with most time left
                        current_time = time.time()
                        try:
                            items_to_copy.sort(
                                key=lambda x: old_cache.timer.get(x[0], 0)
                                - current_time,
                                reverse=True,
                            )
                        except (AttributeError, TypeError):
                            pass  # Fall back to unsorted

                    # Transfer most recent items first, up to new capacity
                    for key, value in items_to_copy[:new_size]:
                        self.memory_cache[key] = value

                self.stats["size"] = len(self.memory_cache)
                logger.info(
                    f"Cache {self.namespace} resized from {len(old_cache)} to {new_size} items"
                )

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple items from the cache at once with optimized deserialization."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_get_many"
            ):
                result = {}
                missing_keys = []

                # First try memory cache for all keys
                for key in keys:
                    if key in self.memory_cache:
                        result[key] = self.memory_cache[key]
                        self.stats["memory_hits"] += 1
                    else:
                        missing_keys.append(key)
                        self.stats["memory_misses"] += 1

                # If disk cache is available, fetch missing keys
                if missing_keys and self.disk_cache and self.disk_enabled:
                    try:
                        # Try bulk retrieval if available (depends on diskcache version)
                        if hasattr(self.disk_cache, "get_many"):
                            with self._track_latency():
                                disk_results = self.disk_cache.get_many(missing_keys)

                                for key, serialized_value in disk_results.items():
                                    if serialized_value is not None:
                                        # Handle serialized format
                                        if (
                                            isinstance(serialized_value, tuple)
                                            and len(serialized_value) == 2
                                        ):
                                            try:
                                                data, format_id = serialized_value
                                                value = self.serializer.deserialize(
                                                    data, format_id
                                                )
                                            except Exception as e:
                                                logger.warning(
                                                    f"Deserialization error for {key}: {e}"
                                                )
                                                continue
                                        else:
                                            # Legacy format
                                            value = serialized_value

                                        # Store raw value in memory
                                        self.memory_cache[key] = value
                                        result[key] = value
                                        self.stats["disk_hits"] += 1
                                    else:
                                        self.stats["disk_misses"] += 1

                                # Mark successful disk operation
                                if disk_results:
                                    self._handle_disk_success()
                        else:
                            # Fall back to individual lookups
                            disk_success = False
                            for key in missing_keys:
                                try:
                                    if key in self.disk_cache:
                                        serialized_value = self.disk_cache[key]

                                        # Handle serialized format
                                        if (
                                            isinstance(serialized_value, tuple)
                                            and len(serialized_value) == 2
                                        ):
                                            try:
                                                data, format_id = serialized_value
                                                value = self.serializer.deserialize(
                                                    data, format_id
                                                )
                                            except Exception as e:
                                                logger.warning(
                                                    f"Deserialization error for {key}: {e}"
                                                )
                                                continue
                                        else:
                                            # Legacy format
                                            value = serialized_value

                                        # Promote to memory cache
                                        self.memory_cache[key] = value
                                        result[key] = value
                                        self.stats["disk_hits"] += 1
                                        disk_success = True
                                    else:
                                        self.stats["disk_misses"] += 1
                                except Exception:
                                    self.stats["disk_misses"] += 1

                            # Mark successful disk operation if at least one key was found
                            if disk_success:
                                self._handle_disk_success()
                    except Exception as e:
                        self._handle_disk_error(f"Disk cache batch read error: {e}")

                # Check memory usage after batch operations
                if len(keys) > 10:  # Only check after significant batch operations
                    self._check_memory_usage()

                return result
        else:
            with self._lock:
                result = {}
                missing_keys = []

                # First try memory cache for all keys
                for key in keys:
                    if key in self.memory_cache:
                        result[key] = self.memory_cache[key]
                        self.stats["memory_hits"] += 1
                    else:
                        missing_keys.append(key)
                        self.stats["memory_misses"] += 1

                # If disk cache is available, fetch missing keys
                if missing_keys and self.disk_cache and self.disk_enabled:
                    try:
                        # Try bulk retrieval if available
                        if hasattr(self.disk_cache, "get_many"):
                            with self._track_latency():
                                disk_results = self.disk_cache.get_many(missing_keys)

                                for key, serialized_value in disk_results.items():
                                    if serialized_value is not None:
                                        # Handle serialized format
                                        if (
                                            isinstance(serialized_value, tuple)
                                            and len(serialized_value) == 2
                                        ):
                                            try:
                                                data, format_id = serialized_value
                                                value = self.serializer.deserialize(
                                                    data, format_id
                                                )
                                            except Exception as e:
                                                logger.warning(
                                                    f"Deserialization error for {key}: {e}"
                                                )
                                                continue
                                        else:
                                            # Legacy format
                                            value = serialized_value

                                        # Store raw value in memory
                                        self.memory_cache[key] = value
                                        result[key] = value
                                        self.stats["disk_hits"] += 1
                                    else:
                                        self.stats["disk_misses"] += 1

                                # Mark successful disk operation
                                if disk_results:
                                    self._handle_disk_success()
                        else:
                            # Fall back to individual lookups
                            disk_success = False
                            for key in missing_keys:
                                try:
                                    if key in self.disk_cache:
                                        serialized_value = self.disk_cache[key]

                                        # Handle serialized format
                                        if (
                                            isinstance(serialized_value, tuple)
                                            and len(serialized_value) == 2
                                        ):
                                            try:
                                                data, format_id = serialized_value
                                                value = self.serializer.deserialize(
                                                    data, format_id
                                                )
                                            except Exception as e:
                                                logger.warning(
                                                    f"Deserialization error for {key}: {e}"
                                                )
                                                continue
                                        else:
                                            # Legacy format
                                            value = serialized_value

                                        # Promote to memory cache
                                        self.memory_cache[key] = value
                                        result[key] = value
                                        self.stats["disk_hits"] += 1
                                        disk_success = True
                                    else:
                                        self.stats["disk_misses"] += 1
                                except Exception:
                                    self.stats["disk_misses"] += 1

                            # Mark successful disk operation if at least one key was found
                            if disk_success:
                                self._handle_disk_success()
                    except Exception as e:
                        self._handle_disk_error(f"Disk cache batch read error: {e}")

                # Check memory usage after batch operations
                if len(keys) > 10:
                    self._check_memory_usage()

                return result

    def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store multiple items in the cache at once with efficient serialization."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_set_many"
            ):
                success = True

                # Check memory usage before batch operation
                self._check_memory_usage()

                # Set all items in memory cache - raw values for performance
                for key, value in items.items():
                    try:
                        self.memory_cache[key] = value  # Store raw value
                    except Exception as e:
                        logger.warning(f"Error setting memory cache item {key}: {e}")
                        success = False

                self.stats["size"] = len(self.memory_cache)

                # Set items in disk cache using efficient serialization
                if self.disk_cache and self.disk_enabled:
                    disk_success = False
                    try:
                        # Prepare serialized items for disk
                        serialized_items = {}
                        for key, value in items.items():
                            try:
                                data, format_id = self.serializer.serialize(value)
                                serialized_items[key] = (data, format_id)
                            except Exception as e:
                                logger.warning(f"Error serializing {key}: {e}")
                                success = False

                        # Try bulk insertion if available
                        if hasattr(self.disk_cache, "set_many"):
                            with self._track_latency():
                                if self.disk_cache.set_many(serialized_items):
                                    disk_success = True
                                else:
                                    success = False

                                # Apply TTL if needed
                                if ttl is not None:
                                    for key in serialized_items:
                                        try:
                                            self.disk_cache.touch(key, ttl)
                                        except Exception:
                                            pass  # Ignore TTL errors
                        else:
                            # Fall back to individual insertions
                            disk_success = False
                            for key, serialized_value in serialized_items.items():
                                try:
                                    self.disk_cache[key] = serialized_value
                                    if ttl is not None:
                                        self.disk_cache.touch(key, ttl)
                                    disk_success = True
                                except Exception as e:
                                    logger.warning(
                                        f"Error setting disk cache item {key}: {e}"
                                    )
                                    success = False

                        # Mark successful disk operation if any items were stored
                        if disk_success:
                            self._handle_disk_success()
                    except Exception as e:
                        self._handle_disk_error(f"Disk cache batch write error: {e}")
                        success = False

                return success
        else:
            with self._lock:
                success = True

                # Check memory usage before batch operation
                self._check_memory_usage()

                # Set all items in memory cache - raw values for performance
                for key, value in items.items():
                    try:
                        self.memory_cache[key] = value  # Store raw value
                    except Exception as e:
                        logger.warning(f"Error setting memory cache item {key}: {e}")
                        success = False

                self.stats["size"] = len(self.memory_cache)

                # Set items in disk cache using efficient serialization
                if self.disk_cache and self.disk_enabled:
                    disk_success = False
                    try:
                        # Prepare serialized items for disk
                        serialized_items = {}
                        for key, value in items.items():
                            try:
                                data, format_id = self.serializer.serialize(value)
                                serialized_items[key] = (data, format_id)
                            except Exception as e:
                                logger.warning(f"Error serializing {key}: {e}")
                                success = False

                        # Try bulk insertion if available
                        if hasattr(self.disk_cache, "set_many"):
                            with self._track_latency():
                                if self.disk_cache.set_many(serialized_items):
                                    disk_success = True
                                else:
                                    success = False

                                # Apply TTL if needed
                                if ttl is not None:
                                    for key in serialized_items:
                                        try:
                                            self.disk_cache.touch(key, ttl)
                                        except Exception:
                                            pass  # Ignore TTL errors
                        else:
                            # Fall back to individual insertions
                            disk_success = False
                            for key, serialized_value in serialized_items.items():
                                try:
                                    self.disk_cache[key] = serialized_value
                                    if ttl is not None:
                                        self.disk_cache.touch(key, ttl)
                                    disk_success = True
                                except Exception as e:
                                    logger.warning(
                                        f"Error setting disk cache item {key}: {e}"
                                    )
                                    success = False

                        # Mark successful disk operation if any items were stored
                        if disk_success:
                            self._handle_disk_success()
                    except Exception as e:
                        self._handle_disk_error(f"Disk cache batch write error: {e}")
                        success = False

                return success

    def _handle_disk_error(self, error_msg: str) -> None:
        """Handle disk cache errors with circuit breaker pattern."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_disk_error"
            ):
                self.stats["disk_errors"] += 1
                self.disk_failures += 1
                self.disk_failure_window.append(time.time())
                logger.warning(
                    f"{error_msg} (failures: {self.disk_failures}/{self.max_failures})"
                )

                # Open circuit if too many failures within a short period
                if self.disk_failures >= self.max_failures:
                    # Check if failures are within a short window (e.g., 5 minutes)
                    window_start = time.time() - 300  # 5 minutes
                    self.disk_failure_window = [
                        t for t in self.disk_failure_window if t >= window_start
                    ]

                    if len(self.disk_failure_window) >= self.max_failures:
                        logger.error(
                            f"Disk cache circuit opened after {self.disk_failures} failures"
                        )
                        self.disk_enabled = False
        else:
            with self._lock:
                self.stats["disk_errors"] += 1
                self.disk_failures += 1
                self.disk_failure_window.append(time.time())
                logger.warning(
                    f"{error_msg} (failures: {self.disk_failures}/{self.max_failures})"
                )

                # Open circuit if too many failures within a short period
                if self.disk_failures >= self.max_failures:
                    # Check if failures are within a short window (e.g., 5 minutes)
                    window_start = time.time() - 300  # 5 minutes
                    self.disk_failure_window = [
                        t for t in self.disk_failure_window if t >= window_start
                    ]

                    if len(self.disk_failure_window) >= self.max_failures:
                        logger.error(
                            f"Disk cache circuit opened after {self.disk_failures} failures"
                        )
                        self.disk_enabled = False

    def _handle_disk_success(self) -> None:
        """
        Reset or reduce failure count after successful disk operations.
        Helps recover from transient failures and re-enable disk cache.
        """
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_disk_success"
            ):
                if self.disk_failures > 0:
                    # Gradual recovery - reduce failure count by 1
                    self.disk_failures = max(0, self.disk_failures - 1)

                    # Clean up old failures from the window
                    window_start = time.time() - 300  # 5 minutes
                    self.disk_failure_window = [
                        t for t in self.disk_failure_window if t >= window_start
                    ]

                    # Re-enable disk cache if it was disabled
                    if not self.disk_enabled and self.disk_failures < self.max_failures:
                        logger.info(
                            "Re-enabling disk cache after successful operations"
                        )
                        self.disk_enabled = True
        else:
            with self._lock:
                if self.disk_failures > 0:
                    # Gradual recovery - reduce failure count by 1
                    self.disk_failures = max(0, self.disk_failures - 1)

                    # Clean up old failures from the window
                    window_start = time.time() - 300  # 5 minutes
                    self.disk_failure_window = [
                        t for t in self.disk_failure_window if t >= window_start
                    ]

                    # Re-enable disk cache if it was disabled
                    if not self.disk_enabled and self.disk_failures < self.max_failures:
                        logger.info(
                            "Re-enabling disk cache after successful operations"
                        )
                        self.disk_enabled = True

    @contextmanager
    def _track_latency(self):
        """Context manager to track operation latency."""
        start = time.perf_counter()
        yield
        latency = time.perf_counter() - start

        # Track latency statistics
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_latency"
            ):
                self.stats["latency_sum"] += latency
                self.stats["latency_count"] += 1
                self.stats["max_latency"] = max(self.stats["max_latency"], latency)

                # Log slow operations (over 100ms)
                if latency > 0.1:
                    logger.debug(
                        f"Slow cache operation in {self.namespace}: {latency:.4f}s"
                    )
        else:
            with self._lock:
                self.stats["latency_sum"] += latency
                self.stats["latency_count"] += 1
                self.stats["max_latency"] = max(self.stats["max_latency"], latency)

                # Log slow operations (over 100ms)
                if latency > 0.1:
                    logger.debug(
                        f"Slow cache operation in {self.namespace}: {latency:.4f}s"
                    )

    def migrate_to_disk(
        self,
        percent: float = 10.0,
        policy: str = "lru",
        item_selector: Optional[
            Callable[[Dict[str, Any]], List[Tuple[str, Any]]]
        ] = None,
    ) -> int:
        """
        Migrate least recently used or selected items from memory to disk cache.

        Args:
            percent: Percentage of memory cache items to migrate
            policy: Migration policy - 'lru' (least recently used), 'size' (largest items first),
                   or 'custom' (using item_selector)
            item_selector: Custom function to select items for migration
                         Should take the memory cache dict and return a list of (key, value) tuples
                         in the order they should be migrated

        Returns:
            int: Number of items migrated
        """
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"hybrid_cache_{self.namespace}_migrate"
            ):
                return self._perform_migration(percent, policy, item_selector)
        else:
            with self._lock:
                return self._perform_migration(percent, policy, item_selector)

    def _perform_migration(
        self,
        percent: float,
        policy: str = "lru",
        item_selector: Optional[
            Callable[[Dict[str, Any]], List[Tuple[str, Any]]]
        ] = None,
    ) -> int:
        """Internal method to perform the actual migration work.

        Args:
            percent: Percentage of memory cache items to migrate
            policy: Migration policy to use
            item_selector: Optional function for custom migration selection

        Returns:
            int: Number of items migrated
        """
        # Skip if disk cache is unavailable
        if not self.disk_cache or not self.disk_enabled:
            return 0

        # Calculate how many items to migrate
        if not self.memory_cache or len(self.memory_cache) < 100:
            return 0

        migrate_count = int(len(self.memory_cache) * percent / 100)
        if migrate_count <= 0:
            return 0

        migrated = 0
        disk_success = False
        items_to_move = {}

        try:
            # Extract items based on selected policy
            if policy == "lru":
                # Original LRU behavior - least recently used items first
                for _ in range(migrate_count):
                    try:
                        key, value = self.memory_cache.popitem()
                        items_to_move[key] = value
                    except KeyError:
                        break  # Cache is empty

            elif policy == "size":
                # Size-based migration - largest items first
                # Process in batches to reduce memory overhead
                batch_size = min(1000, migrate_count)  # Process up to 1000 at once

                def estimate_size(item: Any) -> int:
                    """Estimate the size of an object in bytes."""
                    try:
                        # Handle ML-specific types efficiently
                        if hasattr(item, "nbytes"):  # NumPy arrays, torch tensors
                            return item.nbytes
                        elif hasattr(item, "__sizeof__"):
                            return item.__sizeof__()
                        elif isinstance(item, (bytes, bytearray)):
                            return len(item)
                        elif isinstance(item, str):
                            return len(item) * 2  # Unicode chars use ~2 bytes
                        elif isinstance(item, (list, tuple)):
                            # Sample for large collections
                            sample = item[:20] if len(item) > 20 else item
                            avg = (
                                sum(estimate_size(i) for i in sample) / len(sample)
                                if sample
                                else 0
                            )
                            return int(avg * len(item))
                        elif isinstance(item, dict):
                            # Sample for large dictionaries
                            sample_keys = (
                                list(item.keys())[:20]
                                if len(item) > 20
                                else list(item.keys())
                            )
                            if not sample_keys:
                                return 0
                            avg = sum(
                                estimate_size(item[k]) for k in sample_keys
                            ) / len(sample_keys)
                            return int(avg * len(item))
                        else:
                            import sys

                            return sys.getsizeof(item)
                    except Exception:
                        return 1  # Default size

                processed_keys = set()
                remaining = migrate_count

                while remaining > 0 and len(processed_keys) < len(self.memory_cache):
                    batch = []
                    # Build current batch
                    for key, value in list(
                        self.memory_cache.items()
                    ):  # Use list() to avoid modification during iteration
                        if key not in processed_keys and len(batch) < batch_size:
                            batch.append((key, value, estimate_size(value)))
                            processed_keys.add(key)

                    if not batch:
                        break

                    # Sort batch by estimated size (largest first)
                    batch.sort(key=lambda x: x[2], reverse=True)

                    # Select items from batch
                    to_select = min(remaining, len(batch))
                    for key, value, _ in batch[:to_select]:
                        items_to_move[key] = value
                        if (
                            key in self.memory_cache
                        ):  # Double check the key still exists
                            del self.memory_cache[key]
                        remaining -= 1

            elif policy == "custom" and item_selector is not None:
                # Custom migration policy using provided selector function
                try:
                    # Create a safe view of the cache that won't be modified during selection
                    cache_copy = dict(self.memory_cache)
                    selected_items = item_selector(cache_copy)

                    # Add selected items to migration queue
                    count = 0
                    for key, value in selected_items:
                        if count >= migrate_count:
                            break
                        if key in self.memory_cache:
                            items_to_move[key] = value
                            del self.memory_cache[key]
                            count += 1
                except Exception as e:
                    logger.error(
                        f"Error in custom migration selector: {e}", exc_info=True
                    )
                    # Fall back to LRU policy
                    logger.warning(
                        f"Falling back to LRU policy after custom selector error: {e}"
                    )
                    for _ in range(min(migrate_count, len(self.memory_cache))):
                        try:
                            key, value = self.memory_cache.popitem()
                            items_to_move[key] = value
                        except KeyError:
                            break
            else:
                # Default to LRU if policy is unrecognized
                logger.warning(f"Unrecognized migration policy '{policy}', using LRU")
                for _ in range(migrate_count):
                    try:
                        key, value = self.memory_cache.popitem()
                        items_to_move[key] = value
                    except KeyError:
                        break

            # Store items to disk cache
            if items_to_move:
                for key, value in items_to_move.items():
                    try:
                        # Serialize the value for efficient disk storage
                        with self._track_latency():
                            data, format_id = self.serializer.serialize(value)
                            self.disk_cache[key] = (data, format_id)
                            migrated += 1
                            disk_success = True
                    except Exception as e:
                        logger.warning(f"Error migrating item {key} to disk: {e}")
                        # If we can't move to disk, put it back in memory
                        self.memory_cache[key] = value
                        self._handle_disk_error(f"Migration error for {key}: {e}")

                # Record stats
                self.stats["size"] = len(self.memory_cache)
                self.stats["migrations"] = self.stats.get("migrations", 0) + 1
                self.stats["migrated_items"] = (
                    self.stats.get("migrated_items", 0) + migrated
                )

                # Track policy usage for analytics
                if "migration_policies" not in self.stats:
                    self.stats["migration_policies"] = {}
                policy_stats = self.stats["migration_policies"]
                policy_stats[policy] = policy_stats.get(policy, 0) + 1

                if disk_success:
                    self._handle_disk_success()

                logger.info(
                    f"Migrated {migrated} items from memory to disk cache for {self.namespace} "
                    f"using '{policy}' policy"
                )

            return migrated

        except Exception as e:
            logger.error(f"Error during cache migration: {e}", exc_info=True)
            return 0

    def _check_memory_usage(self):
        """Monitor and optimize memory usage using the centralized function."""

        def trim_callback(memory_usage: float) -> None:
            """Traditional trimming method that simply removes entries"""
            # Now just focused on trimming as migration is handled separately
            if len(self.memory_cache) > 100:
                try:
                    trim_percent = min(
                        30.0, max(10.0, memory_usage - self.max_memory_percent + 10.0)
                    )
                    trim_count = self.trim(trim_percent)
                    if trim_count > 0:
                        logger.info(
                            f"Trimmed {trim_count} items from {self.namespace} cache"
                        )
                except Exception as e:
                    logger.error(f"Error during cache trimming: {e}")

        def migrate_callback(memory_usage: float) -> int:
            """Move items to disk instead of just removing them"""
            # Don't migrate if disk cache isn't available
            if not self.disk_cache or not self.disk_enabled:
                return 0

            # Calculate percentage based on memory pressure
            migrate_percent = min(
                40.0, max(10.0, memory_usage - self.max_memory_percent + 15.0)
            )
            migrated_count = self.migrate_to_disk(migrate_percent)

            # Update stats for migration attempts vs successes
            if not hasattr(self.stats, "migration_attempts"):
                self.stats["migration_attempts"] = 0
                self.stats["migration_successes"] = 0

            self.stats["migration_attempts"] = (
                self.stats.get("migration_attempts", 0) + 1
            )
            if migrated_count > 0:
                self.stats["migration_successes"] = (
                    self.stats.get("migration_successes", 0) + 1
                )

            # Only run trim if migration didn't free enough memory or didn't migrate many items
            # This prevents unnecessary trimming when migration was successful
            memory_still_high = (
                psutil.virtual_memory().percent > self.max_memory_percent - 5
            )
            if migrated_count < 10 and memory_still_high:
                trim_callback(memory_usage)

            return migrated_count

        self.last_memory_check = check_memory_usage(
            f"hybrid_cache_{self.namespace}_{id(self)}",
            self.last_memory_check,
            self.memory_check_interval,
            self.max_memory_percent,
            trim_callback,
            migrate_callback,  # Add migration callback
        )


class LargeObjectCacheBackend(CacheBackend):
    """Cache backend optimized for large objects using weak references."""

    def __init__(self, config: Dict[str, Any], namespace: str = "large_objects"):
        self.memory_cache = {}  # Dict to store weak references
        self.strong_refs = []  # List to optionally hold strong references
        self._lock = RLock()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a large object from the cache."""
        with self._lock:
            ref = self.memory_cache.get(key)
            if ref is not None:
                obj = ref()  # Get the referenced object
                if obj is not None:
                    return obj
                # Remove key if object was garbage collected
                self.memory_cache.pop(key, None)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a large object in the cache using a weak reference."""
        with self._lock:
            self.memory_cache[key] = weakref.ref(value)
            return True

    def add_permanent_object(self, key: str, value: Any) -> None:
        """Add an object that should never be garbage collected."""
        with self._lock:
            self.strong_refs.append(value)  # Keep a strong reference
            self.memory_cache[key] = weakref.ref(value)


class AccessCountingMixin:
    """
    Mixin to track access counts for cache entries to support intelligent promotion policies.

    Classes using this mixin must provide a `contains(key)` method that returns True if
    the key exists in the cache, False otherwise.
    """

    def __init__(self):
        self.access_counts = {}  # Dict to track access counts
        self.promotion_threshold = 3  # Default threshold for promotion
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour between cleanups

    def track_access(self, key: str) -> int:
        """
        Track access to a key and return the updated access count.
        Periodically cleans up access counts for keys that no longer exist.

        Args:
            key: The cache key being accessed

        Returns:
            int: The updated access count
        """
        # Periodically clean up access counts for non-existent keys
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_access_counts()
            self._last_cleanup = current_time

        # Increment and return the access count
        count = self.access_counts.get(key, 0) + 1
        self.access_counts[key] = count
        return count

    def _cleanup_access_counts(self):
        """Remove access counts for keys that no longer exist in the cache."""
        # Check if the required method is available
        if not hasattr(self, "contains") or not callable(getattr(self, "contains")):
            logger.warning(
                "AccessCountingMixin requires 'contains' method but none found"
            )
            return

        keys_to_remove = []
        for key in list(
            self.access_counts.keys()
        ):  # Use list to avoid modification during iteration
            if not self.contains(key):  # This assumes 'contains' method exists
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.access_counts[key]

    def set_promotion_threshold(self, threshold: int):
        """Set the access count threshold for promotion from disk to memory."""
        self.promotion_threshold = max(1, threshold)  # Ensure positive threshold


def secure_cache_directory(
    directory_path: str, fail_if_unsecured: bool = False
) -> bool:
    """
    Apply security measures to the cache directory.

    Args:
        directory_path: Path to the cache directory
        fail_if_unsecured: If True, raises an exception when security cannot be applied

    Returns:
        bool: True if security measures were successfully applied, False otherwise
    """
    if not os.path.exists(directory_path):
        error_msg = f"Cannot secure non-existent directory: {directory_path}"
        logger.error(error_msg)
        if fail_if_unsecured:
            raise ValueError(error_msg)
        return False

    try:
        security_applied = False

        if os.name == "posix":  # Unix/Linux/MacOS
            os.chmod(directory_path, 0o700)  # Only owner can read/write/execute
            logger.info(
                f"Applied POSIX secure permissions to cache directory: {directory_path}"
            )
            security_applied = True
        elif os.name == "nt":  # Windows
            try:
                import win32security  # May fail if not installed
                import ntsecuritycon as con
                import win32api

                # Get current user's SID
                username = win32api.GetUserName()
                sid, _, _ = win32security.LookupAccountName(None, username)

                # Create a security descriptor with owner-only permissions
                sec_desc = win32security.SECURITY_DESCRIPTOR()
                sec_desc.SetSecurityDescriptorOwner(sid, False)

                # Get DACL
                security = win32security.GetFileSecurity(
                    directory_path, win32security.DACL_SECURITY_INFORMATION
                )
                dacl = security.GetSecurityDescriptorDacl()

                # Create a new DACL giving full control to owner only
                dacl = win32security.ACL()
                dacl.AddAccessAllowedAce(
                    win32security.ACL_REVISION, con.FILE_ALL_ACCESS, sid
                )

                # Set the new DACL
                sec_desc.SetSecurityDescriptorDacl(1, dacl, 0)
                win32security.SetFileSecurity(
                    directory_path, win32security.DACL_SECURITY_INFORMATION, sec_desc
                )
                logger.info(
                    f"Applied Windows secure permissions to cache directory: {directory_path}"
                )
                security_applied = True
            except ImportError:
                logger.warning(
                    "Windows security modules not available (pywin32 required)"
                )
                # Try a basic fallback with icacls if available
                try:
                    import subprocess

                    username = os.getlogin()
                    # Use icacls to set permissions (Windows only)
                    subprocess.run(
                        [
                            "icacls",
                            directory_path,
                            "/inheritance:r",
                            "/grant:r",
                            f"{username}:(OI)(CI)F",
                        ],
                        check=True,
                        capture_output=True,
                    )
                    logger.info(
                        f"Applied fallback Windows permissions using icacls to: {directory_path}"
                    )
                    security_applied = True
                except (ImportError, subprocess.SubprocessError) as e:
                    logger.warning(f"Fallback security measures failed: {e}")
        else:
            logger.warning(f"Cannot secure directory on unknown OS type: {os.name}")

        # Verify security was applied
        if not security_applied:
            error_msg = f"Could not apply any security measures to {directory_path}"
            logger.error(error_msg)
            if fail_if_unsecured:
                raise RuntimeError(error_msg)
            return False

        return security_applied

    except (OSError, PermissionError) as e:
        error_msg = f"Security error securing cache directory {directory_path}: {e}"
        logger.error(error_msg)
        if fail_if_unsecured:
            raise RuntimeError(error_msg) from e
        return False


class EnhancedHybridCacheBackend(HybridCacheBackend, AccessCountingMixin):
    """
    Enhanced hybrid cache backend with access counting and security features
    """

    def __init__(self, config: Dict[str, Any], namespace: str = "general"):
        HybridCacheBackend.__init__(self, config, namespace)
        AccessCountingMixin.__init__(self)

        # Apply security to cache directory if available
        if self.disk_cache:
            cache_dir = self.disk_cache.directory
            fail_if_unsecured = config.get("caching", {}).get(
                "secure_cache_required", False
            )
            try:
                security_applied = secure_cache_directory(cache_dir, fail_if_unsecured)
                if not security_applied:
                    logger.warning(
                        f"Cache directory {cache_dir} could not be secured. Cache data may be accessible to other users."
                    )
            except Exception as e:
                if fail_if_unsecured:
                    logger.critical(
                        f"Cache initialization failed due to security requirements: {e}"
                    )
                    raise
                else:
                    logger.error(
                        f"Cache security error (continuing with unsecured cache): {e}"
                    )

    def _perform_migration(self, percent: float) -> int:
        """
        Override to use access counts for more intelligent migration decisions.
        """
        if not self.disk_cache or not self.disk_enabled:
            return 0

        if not self.memory_cache or len(self.memory_cache) < 100:
            return 0

        migrate_count = int(len(self.memory_cache) * percent / 100)
        if migrate_count <= 0:
            return 0

        migrated = 0
        disk_success = False

        try:
            # Use a heap for efficient selection of least accessed items
            import heapq

            # Create a list of (access_count, key) for heap
            candidates = []
            for key in list(self.memory_cache.keys()):
                count = self.access_counts.get(key, 0)
                # Only consider items that haven't been accessed many times
                if count < self.promotion_threshold:
                    candidates.append((count, key))

            # Convert to heap and get the least accessed items
            heapq.heapify(candidates)
            items_to_migrate = []

            while candidates and len(items_to_migrate) < migrate_count:
                count, key = heapq.heappop(candidates)
                # Double check item is still in cache
                if key in self.memory_cache:
                    items_to_migrate.append(key)

            # Process migration
            for key in items_to_migrate:
                try:
                    value = self.memory_cache[key]

                    # Serialize with tracking
                    with self._track_latency():
                        data, format_id = self.serializer.serialize(value)
                        self.disk_cache[key] = (data, format_id)

                        # Remove from memory but keep access count for future priority
                        del self.memory_cache[key]
                        migrated += 1
                        disk_success = True
                except KeyError:
                    # Item was removed in the meantime
                    pass
                except Exception as e:
                    logger.warning(f"Error migrating item {key} to disk: {e}")
                    self._handle_disk_error(f"Migration error for {key}: {e}")

            # Update stats
            if migrated > 0:
                self.stats["size"] = len(self.memory_cache)
                self.stats["migrations"] = self.stats.get("migrations", 0) + 1
                self.stats["migrated_items"] = (
                    self.stats.get("migrated_items", 0) + migrated
                )

                if disk_success:
                    self._handle_disk_success()

                logger.info(
                    f"Migrated {migrated} items from memory to disk for {self.namespace} based on access frequency"
                )

            return migrated

        except Exception as e:
            logger.error(f"Error during frequency-based cache migration: {e}")
            return 0


# -----------------------------------------------------------------------------
# Core Cache Manager
# -----------------------------------------------------------------------------


class CacheDependencyTracker:
    """
    Tracks dependencies between cache entries to enable intelligent invalidation.
    """

    def __init__(self, backend: CacheBackend):
        """Initialize the dependency tracker."""
        self.backend = backend
        self._dependency_key_prefix = "_cache_dep_"

        # Add lock with proper registration if lock_utils is available
        if LOCK_UTILS_AVAILABLE:
            self._lock = create_component_lock("dependency_tracker")
        else:
            self._lock = threading.RLock()

    def register_dependency(self, key: str, depends_on: Union[str, List[str]]) -> None:
        """
        Register that a cache entry depends on another entry or entries.

        Args:
            key: The dependent cache key
            depends_on: The key(s) that this entry depends on
        """
        dependency_list = [depends_on] if isinstance(depends_on, str) else depends_on

        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(self._lock, name="dependency_tracker_register"):
                for dep_key in dependency_list:
                    # Get existing dependents for this key
                    dep_entry_key = f"{self._dependency_key_prefix}{dep_key}"
                    dependents = self.backend.get(dep_entry_key) or set()

                    # Add our key as a dependent
                    if not isinstance(dependents, set):
                        dependents = set(dependents) if dependents else set()

                    dependents.add(key)

                    # Save updated dependents list
                    self.backend.set(dep_entry_key, dependents)
        else:
            with self._lock:
                for dep_key in dependency_list:
                    # Get existing dependents for this key
                    dep_entry_key = f"{self._dependency_key_prefix}{dep_key}"
                    dependents = self.backend.get(dep_entry_key) or set()

                    # Add our key as a dependent
                    if not isinstance(dependents, set):
                        dependents = set(dependents) if dependents else set()

                    dependents.add(key)

                    # Save updated dependents list
                    self.backend.set(dep_entry_key, dependents)

    def invalidate_dependents(self, key: str) -> int:
        """
        Invalidate all cache entries that depend on the specified key.

        Args:
            key: The key whose dependents should be invalidated

        Returns:
            Number of entries invalidated
        """
        dep_entry_key = f"{self._dependency_key_prefix}{key}"
        invalidated = 0

        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(self._lock, name="dependency_tracker_invalidate"):
                # Get dependents
                dependents = self.backend.get(dep_entry_key) or set()
                if not dependents:
                    return 0

                # Invalidate each dependent
                for dependent_key in dependents:
                    # Also invalidate transitive dependencies
                    invalidated += self.invalidate_dependents(dependent_key)

                    # Delete the dependent
                    try:
                        if isinstance(self.backend, HybridCacheBackend):
                            if dependent_key in self.backend.memory_cache:
                                del self.backend.memory_cache[dependent_key]
                                invalidated += 1
                            if self.backend.disk_cache and self.backend.disk_enabled:
                                try:
                                    if dependent_key in self.backend.disk_cache:
                                        del self.backend.disk_cache[dependent_key]
                                        invalidated += 1
                                except Exception as e:
                                    logger.warning(
                                        f"Error invalidating disk cache entry: {e}"
                                    )
                        elif isinstance(self.backend, MemoryCacheBackend):
                            if dependent_key in self.backend.memory_cache:
                                del self.backend.memory_cache[dependent_key]
                                invalidated += 1
                    except Exception as e:
                        logger.warning(
                            f"Error invalidating cache entry {dependent_key}: {e}"
                        )

                # Clear the dependency list itself
                self.backend.set(dep_entry_key, set())

            return invalidated
        else:
            with self._lock:
                # Get dependents
                dependents = self.backend.get(dep_entry_key) or set()
                if not dependents:
                    return 0

                # Invalidate each dependent
                for dependent_key in dependents:
                    # Also invalidate transitive dependencies
                    invalidated += self.invalidate_dependents(dependent_key)

                    # Delete the dependent
                    try:
                        if isinstance(self.backend, HybridCacheBackend):
                            if dependent_key in self.backend.memory_cache:
                                del self.backend.memory_cache[dependent_key]
                                invalidated += 1
                            if self.backend.disk_cache and self.backend.disk_enabled:
                                try:
                                    if dependent_key in self.backend.disk_cache:
                                        del self.backend.disk_cache[dependent_key]
                                        invalidated += 1
                                except Exception as e:
                                    logger.warning(
                                        f"Error invalidating disk cache entry: {e}"
                                    )
                        elif isinstance(self.backend, MemoryCacheBackend):
                            if dependent_key in self.backend.memory_cache:
                                del self.backend.memory_cache[dependent_key]
                                invalidated += 1
                    except Exception as e:
                        logger.warning(
                            f"Error invalidating cache entry {dependent_key}: {e}"
                        )

                # Clear the dependency list itself
                self.backend.set(dep_entry_key, set())

            return invalidated


class CacheManager:
    """
    Unified cache manager with support for different backends and specialized operations.
    """

    def __init__(
        self, config: Dict[str, Any], namespace: str = "general", backend: str = "auto"
    ):
        """
        Initialize a cache manager with the specified backend.

        Args:
            config: Configuration dictionary
            namespace: Cache namespace
            backend: Cache backend type ("memory", "disk", "hybrid", or "auto")
        """
        self.config = config
        self.namespace = namespace
        self.salt = get_cache_salt(config)
        self.config_hash = hash_config_sections(config, self.salt)

        # Determine the best backend if "auto" is specified
        if backend == "auto":
            # Use hybrid if disk cache directory is configured
            if config.get("caching", {}).get("cache_dir"):
                backend = "hybrid"
            else:
                backend = "memory"

        # Initialize the appropriate backend
        if backend == "hybrid":
            self.backend = HybridCacheBackend(config, namespace)
        else:
            self.backend = MemoryCacheBackend(config, namespace)

        logger.debug(
            f"Initialized cache manager for {namespace} using {backend} backend"
        )

    def get(self, key: str) -> Any:
        """Get an item from the cache."""
        cache_key = self._create_key(key)
        return self.backend.get(cache_key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set an item in the cache."""
        cache_key = self._create_key(key)
        return self.backend.set(cache_key, value, ttl)

    def clear(self) -> None:
        """Clear the cache."""
        self.backend.clear()

    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        cache_key = self._create_key(key)
        return self.backend.contains(cache_key)

    def get_stats(self) -> "CacheStats":
        """Get cache statistics."""
        return self.backend.get_stats()

    def trim(self, percent: float = 10.0) -> int:
        """Trim percentage of cache entries."""
        return self.backend.trim(percent)

    def adjust_capacity(self, new_size: int) -> None:
        """Adjust the cache capacity."""
        self.backend.adjust_capacity(new_size)

    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple items from cache efficiently using backend batch capabilities."""
        # Transform all keys to include namespace, version and config hash
        cache_keys = [self._create_key(key) for key in keys]

        # Get results using backend's optimized batch method
        cache_results = self.backend.get_many(cache_keys)

        # Transform the results back to the original key format for the caller
        results = {}
        for i, key in enumerate(keys):
            cache_key = cache_keys[i]
            if cache_key in cache_results:
                results[key] = cache_results[cache_key]

        return results

    def batch_set(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple items in cache efficiently using backend batch capabilities."""
        # Transform the input dictionary to use proper cache keys
        cache_items = {self._create_key(key): value for key, value in items.items()}

        # Use the backend's optimized batch set operation
        return self.backend.set_many(cache_items, ttl)

    def update_config_hash(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the config hash used for cache keys.

        Args:
            config: New configuration (uses stored config if None)
        """
        if config is not None:
            self.config = config
            self.salt = get_cache_salt(config)

        self.config_hash = hash_config_sections(self.config, self.salt)
        logger.debug(
            f"Updated config hash for {self.namespace}: {self.config_hash[:8]}"
        )

    def _create_key(self, key: str) -> str:
        """Create a namespaced and versioned cache key."""
        # Normalize keys that might contain invalid characters
        if isinstance(key, str) and any(
            c in key for c in r'[]{};:"\|<>,./?!@#$%^&*()=+'
        ):
            # Replace problematic characters
            normalized_key = re.sub(r"[^\w\s-]", "_", key)
            # Remove multiple underscores
            normalized_key = re.sub(r"_+", "_", normalized_key)
            key = normalized_key
        return f"{CACHE_VERSION}:{self.namespace}:{self.config_hash[:8]}:{key}"

    def invalidate_by_config_change(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
        sections: Optional[List[str]] = None,
    ) -> Set[str]:
        """
        Selectively invalidate cache based on specific configuration changes.
        Uses granular detection from cache_invalidation module if available.

        Args:
            old_config: Previous configuration
            new_config: New configuration
            sections: Specific sections to check for changes (if None, checks all relevant sections)

        Returns:
            Set of invalidation reasons for logging
        """
        invalidation_reasons = set()

        # Try to use the more advanced config change detector if available
        try:
            from cache_invalidation import ConfigChangeDetector

            # Use the advanced detection method
            detector = ConfigChangeDetector()
            changed_sections, affected_types = detector.detect_changes(
                old_config, new_config
            )

            # Check if critical changes were detected
            if "all" in affected_types:
                logger.info(
                    "Critical configuration changes detected, clearing entire cache"
                )
                self.clear()
                self.config = new_config
                self.update_config_hash()
                return {"critical_changes"}

            # Process section changes
            invalidation_reasons.update(changed_sections)

            # Apply targeted invalidation for specific cache types
            for affected_type in affected_types:
                pattern = f".*:{affected_type}:.*"
                try:
                    self.invalidate_by_pattern(pattern)
                    logger.debug(f"Invalidated entries matching pattern: {pattern}")
                except ValueError as e:
                    logger.warning(
                        f"Skipping invalidation for pattern '{pattern}': {e}"
                    )

        except ImportError:
            # Fall back to simpler section-based comparison
            logger.debug(
                "cache_invalidation module not available, using basic comparison"
            )
            sections_to_check = sections or [
                "stop_words",
                "stop_words_add",
                "stop_words_exclude",
                "text_processing",
                "caching",
                "validation",
                "keyword_categories",
                "vectorization",
            ]

            # Check each section for changes
            for section in sections_to_check:
                old_section = old_config.get(section, {})
                new_section = new_config.get(section, {})

                # Quick check if the sections are identical
                if old_section == new_section:
                    continue

                # Calculate section-specific hashes for more granular comparison
                old_hash = xxhash.xxh3_64(
                    json.dumps(old_section, sort_keys=True).encode()
                ).hexdigest()
                new_hash = xxhash.xxh3_64(
                    json.dumps(new_section, sort_keys=True).encode()
                ).hexdigest()

                if old_hash != new_hash:
                    invalidation_reasons.add(section)

                    # Check for critical paths that require targeted invalidation
                    if section == "vectorization" and old_section.get(
                        "model_name"
                    ) != new_section.get("model_name"):
                        pattern = ".*:vector:.*"
                        try:
                            self.invalidate_by_pattern(pattern)
                            logger.info(
                                f"Model changed, invalidating vector cache entries"
                            )
                        except ValueError as e:
                            logger.warning(f"Skipping vector invalidation: {e}")

        # Only update config hash if relevant changes were detected
        if invalidation_reasons:
            self.config = new_config
            self.update_config_hash()
            logger.info(
                f"Cache {self.namespace} invalidated due to changes in: {', '.join(invalidation_reasons)}"
            )

        return invalidation_reasons

    def set_with_dependencies(
        self,
        key: str,
        value: Any,
        depends_on: Union[str, List[str], None] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set an item in the cache and register its dependencies.

        Args:
            key: Cache key
            value: Value to store
            depends_on: Key(s) that this entry depends on
            ttl: Optional TTL in seconds

        Returns:
            bool: Success status
        """
        cache_key = self._create_key(key)
        success = self.backend.set(cache_key, value, ttl)

        if success and depends_on and hasattr(self, "dependency_tracker"):
            # Convert single key to list for uniform handling
            dep_keys = [depends_on] if isinstance(depends_on, str) else depends_on
            # Create proper cache keys for dependencies
            dep_cache_keys = [self._create_key(dep_key) for dep_key in dep_keys]
            # Register dependencies
            self.dependency_tracker.register_dependency(cache_key, dep_cache_keys)

        return success

    def invalidate_key(self, key: str) -> int:
        """
        Invalidate a specific cache key and its dependents.

        Args:
            key: The key to invalidate

        Returns:
            int: Number of entries invalidated
        """
        cache_key = self._create_key(key)
        invalidated = 0

        # First, invalidate dependents if dependency tracker exists
        if hasattr(self, "dependency_tracker"):
            invalidated += self.dependency_tracker.invalidate_dependents(cache_key)

        # Then invalidate the key itself
        if isinstance(self.backend, HybridCacheBackend):
            # Handle memory cache
            if cache_key in self.backend.memory_cache:
                del self.backend.memory_cache[cache_key]
                invalidated += 1

            # Handle disk cache if available
            if self.backend.disk_cache and self.backend.disk_enabled:
                try:
                    if cache_key in self.backend.disk_cache:
                        del self.backend.disk_cache[cache_key]
                        invalidated += 1
                except Exception as e:
                    logger.warning(f"Error invalidating disk cache entry: {e}")
        elif isinstance(self.backend, MemoryCacheBackend):
            if cache_key in self.backend.memory_cache:
                del self.backend.memory_cache[cache_key]
                invalidated += 1

        return invalidated

    def invalidate_by_pattern(
        self,
        pattern: str,
        batch_size: Optional[int] = None,
        max_invalidations: Optional[int] = None,
    ) -> int:
        """
        Invalidate cache entries that match a specific pattern with memory efficiency.

        Args:
            pattern: Regular expression pattern to match against keys
            batch_size: Number of keys to process in each batch (defaults to config)
            max_invalidations: Maximum number of entries to invalidate (None for no limit)

        Returns:
            int: Number of entries invalidated
        """
        # Validate pattern to prevent accidental deletion of entire cache
        if not pattern or pattern == ".*" or pattern == "*":
            raise ValueError("Pattern too broad - would invalidate entire cache")

        # Use configured values if not specified
        if batch_size is None:
            batch_size = (
                self.config.get("caching", {})
                .get("invalidation", {})
                .get("batch_size", 500)
            )
        if max_invalidations is None:
            max_invalidations = (
                self.config.get("caching", {})
                .get("invalidation", {})
                .get("max_invalidations", 10000)
            )

        try:
            regex = re.compile(pattern)
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            raise ValueError(f"Invalid regex pattern: {e}")

        invalidated = 0

        # Process memory cache entries with appropriate locking
        if isinstance(self.backend, (MemoryCacheBackend, HybridCacheBackend)):
            if LOCK_UTILS_AVAILABLE:
                with coordinated_lock(
                    self._lock, name=f"{self.namespace}_invalidate_pattern"
                ):
                    invalidated += self._process_memory_cache_patterns(
                        regex, batch_size, max_invalidations
                    )
            else:
                with self._lock:
                    invalidated += self._process_memory_cache_patterns(
                        regex, batch_size, max_invalidations
                    )

            # Early return if reached limit
            if max_invalidations is not None and invalidated >= max_invalidations:
                return invalidated

        # Process disk cache if available
        if (
            isinstance(self.backend, HybridCacheBackend)
            and self.backend.disk_cache
            and self.backend.disk_enabled
        ):
            if LOCK_UTILS_AVAILABLE:
                with coordinated_lock(
                    self._lock, name=f"{self.namespace}_invalidate_pattern_disk"
                ):
                    invalidated += self._process_disk_cache_patterns(
                        regex, batch_size, max_invalidations, invalidated
                    )
            else:
                with self._lock:
                    invalidated += self._process_disk_cache_patterns(
                        regex, batch_size, max_invalidations, invalidated
                    )

        logger.info(f"Invalidated {invalidated} entries matching pattern '{pattern}'")
        return invalidated

    def _process_memory_cache_patterns(
        self, regex, batch_size: int, max_invalidations: Optional[int]
    ) -> int:
        """Helper method to process memory cache entries matching a pattern."""
        invalidated = 0
        # Get a copy of keys to avoid modification during iteration
        memory_keys = list(self.backend.memory_cache.keys())

        # Process in batches to control memory usage
        for i in range(0, len(memory_keys), batch_size):
            batch = memory_keys[i : i + batch_size]

            for key in batch:
                if regex.search(key):
                    try:
                        # Process dependencies if available
                        if hasattr(self, "dependency_tracker"):
                            invalidated += (
                                self.dependency_tracker.invalidate_dependents(key)
                            )

                        # Invalidate the key itself
                        del self.backend.memory_cache[key]
                        invalidated += 1

                        # Check limit
                        if (
                            max_invalidations is not None
                            and invalidated >= max_invalidations
                        ):
                            logger.warning(
                                f"Reached max_invalidations limit ({max_invalidations})"
                            )
                            return invalidated
                    except Exception as e:
                        logger.warning(
                            f"Error invalidating memory cache entry {key}: {e}"
                        )

        return invalidated

    def _process_disk_cache_patterns(
        self,
        regex,
        batch_size: int,
        max_invalidations: Optional[int],
        current_invalidated: int,
    ) -> int:
        """Helper method to process disk cache entries matching a pattern."""
        invalidated = 0
        try:
            # Check if transaction support is available
            has_transaction = hasattr(self.backend.disk_cache, "transact")

            # Always check for dependency tracker on each call
            has_dependency_tracker = hasattr(self, "dependency_tracker")

            # Always process keys individually if dependency tracking is available
            # to ensure proper invalidation of dependent keys
            if has_dependency_tracker:
                # With dependency tracking: process keys individually to handle dependencies
                processed = 0
                for key in self.backend.disk_cache.iterkeys():
                    processed += 1
                    key_str = str(key)

                    if regex.search(key_str):
                        try:
                            # First handle dependencies - recheck dependency tracker to be safe
                            if hasattr(self, "dependency_tracker"):
                                dep_invalidated = (
                                    self.dependency_tracker.invalidate_dependents(
                                        key_str
                                    )
                                )
                                invalidated += dep_invalidated

                            # Then delete the key itself
                            try:
                                del self.backend.disk_cache[key]
                                invalidated += 1
                            except Exception as e:
                                logger.warning(
                                    f"Error deleting disk cache key {key}: {e}"
                                )

                            # Check invalidation limit
                            if (
                                max_invalidations is not None
                                and (current_invalidated + invalidated)
                                >= max_invalidations
                            ):
                                logger.warning(
                                    f"Reached max_invalidations limit ({max_invalidations})"
                                )
                                return invalidated
                        except Exception as e:
                            logger.warning(
                                f"Error handling dependency for disk key {key}: {e}"
                            )

                    # Periodically yield control to other threads
                    if processed % 5000 == 0:
                        logger.debug(
                            f"Processed {processed} disk cache keys, invalidated {invalidated} so far"
                        )
            else:
                # No dependency tracker - use batched approach for efficiency
                # This is faster but cannot handle dependencies
                keys_to_delete = []
                processed = 0

                # Iterate through keys efficiently
                for key in self.backend.disk_cache.iterkeys():
                    processed += 1
                    key_str = str(key)

                    if regex.search(key_str):
                        keys_to_delete.append(key)

                    # When batch is full or it's the end, process the batch
                    if len(keys_to_delete) >= batch_size:
                        invalidated += self._delete_disk_keys_batch(
                            keys_to_delete, has_transaction
                        )
                        keys_to_delete = []

                        # Check limit after each batch
                        if (
                            max_invalidations is not None
                            and (current_invalidated + invalidated) >= max_invalidations
                        ):
                            logger.warning(
                                f"Reached max_invalidations limit ({max_invalidations})"
                            )
                            return invalidated

                    # Periodically yield control to other threads when processing many keys
                    if processed % 5000 == 0:
                        logger.debug(
                            f"Processed {processed} disk cache keys, invalidated {invalidated} so far"
                        )

                # Process any remaining keys
                if keys_to_delete:
                    invalidated += self._delete_disk_keys_batch(
                        keys_to_delete, has_transaction
                    )

        except Exception as e:
            logger.error(f"Error processing disk cache: {e}")

        return invalidated

    def _delete_disk_keys_batch(self, keys: List, use_transaction: bool) -> int:
        """Delete a batch of keys from disk cache, optionally using transactions."""
        deleted = 0
        try:
            if use_transaction:
                with self.backend.disk_cache.transact():
                    for key in keys:
                        try:
                            del self.backend.disk_cache[key]
                            deleted += 1
                        except Exception as e:
                            logger.warning(f"Error deleting disk cache key {key}: {e}")
            else:
                # Without transaction support
                for key in keys:
                    try:
                        del self.backend.disk_cache[key]
                        deleted += 1
                    except Exception as e:
                        logger.warning(f"Error deleting disk cache key {key}: {e}")
        except Exception as e:
            logger.error(f"Error in batch deletion: {e}")

        return deleted


class EnhancedCacheManager(CacheManager):
    """
    Enhanced cache manager that uses the improved backend
    """

    def __init__(
        self,
        config: Dict[str, Any],
        namespace: str = "general",
        backend: str = "enhanced",
    ):
        """
        Initialize a cache manager with enhanced backend.

        Args:
            config: Configuration dictionary
            namespace: Cache namespace
            backend: Cache backend type
        """
        self.config = config
        self.namespace = namespace

        # Always use enhanced backend
        self.backend = EnhancedHybridCacheBackend(config, namespace)

        logger.debug(f"Initialized enhanced cache manager for {namespace}")

        # Rest of initialization from parent class
        self.salt = get_cache_salt(config)
        self.config_hash = hash_config_sections(config, self.salt)

        self.dependency_tracker = CacheDependencyTracker(self.backend)

        # Track configuration sections that affect specific cache types
        self.section_dependencies = {
            "text_processing": ["preprocess", "tokenize", "ngram"],
            "stop_words": ["preprocess", "tokenize"],
            "stop_words_add": ["preprocess", "tokenize"],
            "stop_words_exclude": ["preprocess", "tokenize"],
            "vectorization": ["vector"],
            "validation": ["validation"],
            "keyword_categories": ["validation", "keywords"],
            "caching": ["all"],
        }

        logger.debug(
            f"Initialized enhanced cache manager with dependency tracking for {namespace}"
        )

    def invalidate_by_section(self, section_names: List[str]) -> int:
        """
        Invalidate cache entries affected by changes to specified config sections.

        Args:
            section_names: List of configuration section names that changed

        Returns:
            int: Number of cache entries invalidated
        """
        invalidated = 0
        affected_cache_types = set()

        # Determine which cache types are affected by the changed sections
        for section in section_names:
            if section in self.section_dependencies:
                if "all" in self.section_dependencies[section]:
                    # If section affects all cache types, clear everything
                    self.clear()
                    return 1  # Simplified count for total invalidation

                affected_cache_types.update(self.section_dependencies[section])

        # Update config hash to ensure new entries use the new hash
        self.update_config_hash()

        # Log the invalidation
        if affected_cache_types:
            affected_str = ", ".join(sorted(affected_cache_types))
            logger.info(
                f"Invalidating {self.namespace} cache entries for types: {affected_str}"
            )

            # Invalidate entries based on affected types
            for cache_type in affected_cache_types:
                pattern = f".*:{cache_type}:.*"
                type_invalidated = self.invalidate_by_pattern(pattern)
                invalidated += type_invalidated
                logger.debug(
                    f"Invalidated {type_invalidated} entries for type {cache_type}"
                )

        return invalidated


# -----------------------------------------------------------------------------
# Specialized Extensions
# -----------------------------------------------------------------------------


class VectorCacheManager:
    """Specialized cache for vector operations with dimensionality awareness."""

    def __init__(self, config: Dict[str, Any], nlp, namespace: str = "vectorization"):
        # Use the centralized CacheManager instead of reimplementing
        self.cache_manager = CacheManager(config, namespace)
        self.config = config
        self.nlp = nlp
        self.vector_dim = self._get_vector_dimension()
        # Add lock with proper registration
        if LOCK_UTILS_AVAILABLE:
            self._lock = create_component_lock(f"vector_cache_{namespace}")
        else:
            self._lock = threading.RLock()
        # Initialize memory monitor for adaptive caching
        self._last_memory_check = time.time()
        self._memory_check_interval = config.get("caching", {}).get(
            "memory_check_interval", 300
        )
        self._max_memory_percent = config.get("hardware_limits", {}).get(
            "max_ram_usage_percent", 80
        )
        # Use centralized function for cache sizing
        self._base_cache_size = calculate_optimal_cache_size(config, "vector")

        # Vector-specific statistics
        self.stats = {
            "vector_calcs": 0,
            "batch_ops": 0,
            "similarity_calcs": 0,
            "batch_similarity": 0,
            "gpu_ops": 0,
            "gpu_fallbacks": 0,
        }

        # Check GPU availability once during initialization
        self.gpu_available = hasattr(torch, "cuda") and torch.cuda.is_available()
        if self.gpu_available:
            logger.info("GPU acceleration available for vector operations")

        # Add timestamp tracking for invalidation
        self._invalidation_key = self.cache_manager._create_key(
            "_vector_invalidation_timestamp"
        )
        self._init_invalidation_timestamp()

        # Implement a sharded lock system to reduce contention
        vector_config = self.config.get("caching", {}).get("vector_cache", {})
        self._shard_count = vector_config.get("lock_shards", 64)
        self._term_locks = [{} for _ in range(self._shard_count)]

        # Create shard locks with proper integration to lock_utils if available
        if LOCK_UTILS_AVAILABLE:
            self._shard_locks = [
                create_component_lock(f"vector_cache_shard_{namespace}_{i}")
                for i in range(self._shard_count)
            ]
        else:
            self._shard_locks = [threading.RLock() for _ in range(self._shard_count)]

        # Configure lock cleanup
        self._lock_timeout = vector_config.get("lock_timeout", 30)
        self._cleanup_interval = vector_config.get("cleanup_interval", 300)
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = None

        # Start cleanup thread if enabled
        if vector_config.get("enable_lock_cleanup", True):
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_locks,
                daemon=True,
                name=f"VectorCacheLockCleanup_{namespace}",
            )
            self._cleanup_thread.start()
            logger.debug(f"Started vector cache lock cleanup thread for {namespace}")

        # ... rest of existing initialization ...

    def _init_invalidation_timestamp(self) -> None:
        """Initialize or retrieve the invalidation timestamp."""
        current_time = time.time()
        # Try to get existing timestamp
        timestamp = self.cache_manager.backend.get(self._invalidation_key)
        if timestamp is None:
            # Set initial timestamp
            self.cache_manager.backend.set(self._invalidation_key, current_time)
            self._invalidation_timestamp = current_time
        else:
            self._invalidation_timestamp = timestamp

    def get_vector(self, term: str) -> Optional[VectorType]:
        """Get vector for term with caching - minimizing lock scope and checking invalidation time."""
        # Check memory periodically - this should be thread-safe
        self._check_and_adjust_memory()

        # Calculate key and shard
        cache_key = self.cache_manager._create_key(term)
        shard_id = hash(term) % self._shard_count

        # First check if the vector is already being calculated by another thread
        with self._get_shard_lock(shard_id):
            if term in self._term_locks[shard_id]:
                lock_time = self._term_locks[shard_id][term]
                if time.time() - lock_time < self._lock_timeout:
                    # Another thread is calculating this vector, wait a bit
                    time.sleep(0.01)  # Brief pause to avoid spinning
                    return self.get_vector(term)  # Recursive retry
                else:
                    # Stale lock, remove it
                    del self._term_locks[shard_id][term]

        # Try to get vector from cache with metadata key
        vector = self.cache_manager.backend.get(cache_key)
        metadata_key = self.cache_manager._create_key(f"{term}:metadata")
        vector_timestamp = self.cache_manager.backend.get(metadata_key)

        # Check if vector exists and is valid
        if vector is not None and vector_timestamp is not None:
            if vector_timestamp >= self._invalidation_timestamp:
                # Valid vector, return it
                if self.gpu_available:
                    try:
                        return torch.tensor(vector, device="cuda")
                    except Exception as e:
                        logger.debug(
                            f"Failed to create GPU tensor: {e}, falling back to numpy"
                        )
                return np.array(vector)

        # Vector not found or invalid, calculate it
        # First set lock to prevent other threads from calculating the same vector
        with self._get_shard_lock(shard_id):
            # Double check in case another thread just calculated it
            if term in self._term_locks[shard_id]:
                return self.get_vector(term)  # Retry

            # Set lock
            self._term_locks[shard_id][term] = time.time()

        try:
            # Calculate vector
            vec = self._calculate_and_cache_vector(term)

            # Store timestamp separately to maintain backward compatibility
            current_time = time.time()
            self.cache_manager.backend.set(metadata_key, current_time)

            return vec
        finally:
            # Always release lock
            with self._get_shard_lock(shard_id):
                self._term_locks[shard_id].pop(term, None)

    def _get_shard_lock(self, shard_id: int) -> Union[threading.RLock, ContextManager]:
        """Get appropriate lock for a shard, using coordinated_lock if available."""
        lock = self._shard_locks[shard_id]
        if LOCK_UTILS_AVAILABLE:
            return coordinated_lock(
                lock,
                name=f"vector_cache_shard_{self.namespace}_{shard_id}",
                timeout=self._lock_timeout,
            )
        return lock

    def get_vectors_batch(self, terms: List[str], batch_size: int = 50) -> VectorDict:
        """Get vectors for multiple terms with enhanced memory management and adaptive batch sizing."""
        result = {}

        # Track batch operation in stats, use atomic update
        self.stats["batch_ops"] = self.stats.get("batch_ops", 0) + 1

        # Process unique terms only to avoid redundant work
        unique_terms = list(set(terms))

        # First check cache for all terms without locking
        for term in unique_terms:
            cache_key = self.cache_manager._create_key(term)
            metadata_key = self.cache_manager._create_key(f"{term}:metadata")

            vector = self.cache_manager.backend.get(cache_key)
            vector_timestamp = self.cache_manager.backend.get(metadata_key)

            if vector is not None and (
                vector_timestamp is None
                or vector_timestamp >= self._invalidation_timestamp
            ):
                result[term] = np.array(vector)

        # Find missing terms
        missing = [term for term in unique_terms if term not in result]
        if not missing:
            return result

        # Calculate adaptive batch size
        batch_size = self.adaptive_vector_batch_size()

        # Process missing terms in batches with vector calculation
        with self.memory_optimized_operation(f"vector batch ({len(missing)} items)"):
            batches = [
                missing[i : i + batch_size] for i in range(0, len(missing), batch_size)
            ]

            # Get reference to NLP model outside of any locks
            nlp_processor = self.nlp

            for batch in batches:
                # Acquire locks for each term in batch
                terms_to_calculate = []
                for term in batch:
                    shard_id = hash(term) % self._shard_count
                    with self._get_shard_lock(shard_id):
                        if term in self._term_locks[shard_id]:
                            continue  # Skip, already being calculated
                        self._term_locks[shard_id][term] = time.time()
                        terms_to_calculate.append((term, shard_id))

                try:
                    if terms_to_calculate:
                        # Extract just the terms for processing
                        terms_only = [t[0] for t in terms_to_calculate]

                        try:
                            # Process batch through NLP pipeline
                            docs = list(nlp_processor.pipe(terms_only))

                            # Store vectors and update cache with timestamps
                            current_time = time.time()
                            for (term, _), doc in zip(terms_to_calculate, docs):
                                if doc.has_vector:
                                    vec = doc.vector
                                    cache_key = self.cache_manager._create_key(term)
                                    metadata_key = self.cache_manager._create_key(
                                        f"{term}:metadata"
                                    )

                                    # Store vector and its timestamp
                                    self.cache_manager.backend.set(
                                        cache_key, vec.tolist()
                                    )
                                    self.cache_manager.backend.set(
                                        metadata_key, current_time
                                    )

                                    result[term] = vec
                                    self.stats["vector_calcs"] += 1
                        except RuntimeError as e:
                            # Handle CUDA OOM errors with fallback
                            if "CUDA out of memory" in str(e) and batch_size > 5:
                                logger.warning(
                                    f"CUDA OOM error: {e}. Reducing batch size and retrying."
                                )
                                # Recursively retry with smaller batch size
                                smaller_batch_size = max(1, batch_size // 2)

                                # Clear GPU memory before retry
                                if self.gpu_available:
                                    optimize_gpu_memory()

                                # Process this batch with smaller size
                                for mini_batch in [
                                    batch[i : i + smaller_batch_size]
                                    for i in range(0, len(batch), smaller_batch_size)
                                ]:
                                    smaller_result = self.get_vectors_batch(
                                        mini_batch, smaller_batch_size
                                    )
                                    result.update(smaller_result)
                            else:
                                logger.warning(f"Batch vectorization error: {e}")

                                # Fall back to individual processing when batch fails
                                for term in batch:
                                    try:
                                        if term not in result:  # Don't reprocess
                                            vec = self._calculate_and_cache_vector(term)
                                            if vec is not None:
                                                result[term] = vec
                                    except Exception as e2:
                                        logger.error(
                                            f"Vector calculation failed for '{term}': {e2}"
                                        )
                finally:
                    # Release all locks
                    for term, shard_id in terms_to_calculate:
                        with self._get_shard_lock(shard_id):
                            self._term_locks[shard_id].pop(term, None)

        return result

    def invalidate_all(self) -> None:
        """Invalidate all vectors by updating the invalidation timestamp."""
        current_time = time.time()
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"vector_cache_{self.namespace}_invalidate"
            ):
                self.cache_manager.backend.set(self._invalidation_key, current_time)
                self._invalidation_timestamp = current_time
        else:
            with self._lock:
                self.cache_manager.backend.set(self._invalidation_key, current_time)
                self._invalidation_timestamp = current_time

        logger.info(f"Invalidated all vector cache entries at {current_time}")

    def _cleanup_locks(self) -> None:
        """Periodically clean up stale locks."""
        while not self._stop_cleanup.is_set():
            try:
                # Sleep first to avoid cleaning immediately after start
                self._stop_cleanup.wait(self._cleanup_interval)
                if self._stop_cleanup.is_set():
                    break

                current_time = time.time()
                cleaned = 0

                # Clean each shard
                for shard_id in range(self._shard_count):
                    with self._get_shard_lock(shard_id):
                        # Find stale locks
                        stale_terms = [
                            term
                            for term, lock_time in self._term_locks[shard_id].items()
                            if current_time - lock_time > self._lock_timeout
                        ]

                        # Remove stale locks
                        for term in stale_terms:
                            del self._term_locks[shard_id][term]
                            cleaned += 1

                if cleaned > 0:
                    logger.debug(f"Cleaned up {cleaned} stale vector calculation locks")

            except Exception as e:
                logger.error(f"Error in vector cache lock cleanup: {e}")

    def __del__(self) -> None:
        """Clean up resources when the manager is destroyed."""
        try:
            if hasattr(self, "_stop_cleanup") and hasattr(self, "_cleanup_thread"):
                self._stop_cleanup.set()
                if self._cleanup_thread and self._cleanup_thread.is_alive():
                    self._cleanup_thread.join(timeout=1.0)
        except Exception:
            pass  # Ignore errors during cleanup

    # ... existing methods remain unchanged ...

    def calculate_similarity_matrix(
        self, terms: List[str], candidates: List[str]
    ) -> "SimilarityMatrix":
        """
        Calculate the similarity matrix between two lists of terms.
        Much more efficient than calculating individual similarities.
        Uses GPU acceleration when available, with fallback to memory-efficient CPU implementation.

        Args:
            terms: List of query terms
            candidates: List of candidate terms to compare against

        Returns:
            np.ndarray: Similarity matrix of shape (len(terms), len(candidates))
        """
        # Track this operation for statistics
        self.stats["similarity_calcs"] += 1
        self.stats["batch_similarity"] = self.stats.get("batch_similarity", 0) + 1

        # Get all unique terms to avoid redundant vector calculations
        all_terms = list(set(terms + candidates))

        # Get vectors in a single batch operation
        vectors = self.get_vectors_batch(all_terms)

        # Create term to index mapping for fast lookups
        term_to_idx = {term: i for i, term in enumerate(all_terms) if term in vectors}

        # Create arrays for matrix calculation
        term_vectors = []
        for t in terms:
            if t in term_to_idx:
                vec = vectors.get(t)
                if vec is not None:
                    term_vectors.append(vec)
                else:
                    term_vectors.append(np.zeros(self.vector_dim))
            else:
                term_vectors.append(np.zeros(self.vector_dim))

        cand_vectors = []
        for c in candidates:
            if c in term_to_idx:
                vec = vectors.get(c)
                if vec is not None:
                    cand_vectors.append(vec)
                else:
                    cand_vectors.append(np.zeros(self.vector_dim))
            else:
                cand_vectors.append(np.zeros(self.vector_dim))

        # Check if we're dealing with a large matrix that might need special handling
        matrix_size = len(terms) * len(candidates)
        is_large_matrix = matrix_size > 10000
        is_very_large_matrix = matrix_size > 100000

        # Monitor memory before heavy computation
        if is_large_matrix:
            try:
                mem_available = psutil.virtual_memory().available / (1024**2)  # MB
                logger.debug(
                    f"Memory available before similarity calculation: {mem_available:.1f}MB"
                )

                # For extremely large matrices, we might need to force lower precision
                force_lower_precision = mem_available < 1000 and is_very_large_matrix
            except Exception:
                force_lower_precision = False
        else:
            force_lower_precision = False

        # Try to use GPU acceleration if available
        if self.gpu_available:
            try:
                # Convert to torch tensors
                term_matrix = torch.tensor(
                    [
                        v.tolist() if isinstance(v, np.ndarray) else v
                        for v in term_vectors
                    ],
                    device="cuda",
                    # Use lower precision for very large matrices
                    dtype=torch.float16 if force_lower_precision else torch.float32,
                )

                cand_matrix = torch.tensor(
                    [
                        v.tolist() if isinstance(v, np.ndarray) else v
                        for v in cand_vectors
                    ],
                    device="cuda",
                    # Use lower precision for very large matrices
                    dtype=torch.float16 if force_lower_precision else torch.float32,
                )

                # Use chunking for very large matrices to avoid OOM
                if is_very_large_matrix and term_matrix.shape[0] > 1000:
                    chunk_size = min(1000, term_matrix.shape[0] // 2)
                    logger.debug(
                        f"Using chunked GPU processing with chunk size {chunk_size}"
                    )

                    similarity_matrix = torch.zeros(
                        (term_matrix.shape[0], cand_matrix.shape[0]),
                        device="cuda",
                        dtype=torch.float32
                        if not force_lower_precision
                        else torch.float16,
                    )

                    # Process chunks
                    for i in range(0, term_matrix.shape[0], chunk_size):
                        end_i = min(i + chunk_size, term_matrix.shape[0])
                        chunk = term_matrix[i:end_i]

                        # Normalize chunk vectors
                        chunk_norms = torch.norm(chunk, dim=1, keepdim=True)
                        chunk_norms[chunk_norms == 0] = 1
                        chunk_norm = chunk / chunk_norms

                        # Normalize candidate vectors (all at once)
                        if i == 0:  # Only do this once
                            cand_norms = torch.norm(cand_matrix, dim=1, keepdim=True)
                            cand_norms[cand_norms == 0] = 1
                            cand_matrix_norm = cand_matrix / cand_norms

                        # Calculate similarity for this chunk
                        similarity_matrix[i:end_i] = torch.mm(
                            chunk_norm, cand_matrix_norm.t()
                        )

                        # Free memory
                        del chunk, chunk_norms, chunk_norm
                        torch.cuda.empty_cache()
                else:
                    # Normalize the vectors
                    term_norms = torch.norm(term_matrix, dim=1, keepdim=True)
                    cand_norms = torch.norm(cand_matrix, dim=1, keepdim=True)

                    # Replace zero norms with 1 to avoid division by zero
                    term_norms[term_norms == 0] = 1
                    cand_norms[term_norms == 0] = 1

                    # Normalize vectors
                    term_matrix = term_matrix / term_norms
                    cand_matrix = cand_matrix / cand_norms

                    # Calculate similarity matrix in one operation (much faster)
                    similarity_matrix = torch.mm(term_matrix, cand_matrix.t())

                # Ensure values are in [0, 1] range and convert back to numpy
                similarity_matrix = torch.clamp(similarity_matrix, 0, 1).cpu().numpy()

                self.stats["gpu_ops"] = self.stats.get("gpu_ops", 0) + 1

                # Add GPU memory optimization after large matrix calculations
                if is_large_matrix:  # Do this for any large matrix
                    optimize_gpu_memory()

                return similarity_matrix

            except Exception as e:
                logger.warning(f"GPU calculation failed, falling back to CPU: {e}")
                self.stats["gpu_fallbacks"] = self.stats.get("gpu_fallbacks", 0) + 1
                # Fall through to CPU implementation

        # CPU implementation - use memory-efficient algorithms for large matrices
        # Convert to numpy arrays
        term_matrix = np.array(term_vectors)
        cand_matrix = np.array(cand_vectors)

        # Use the optimized CPU implementation based on matrix size
        if is_very_large_matrix:
            # For very large matrices use the most memory-efficient approach
            logger.debug("Using highly optimized similarity for very large matrix")
            precision = "float16" if force_lower_precision else "float32"
            similarity_matrix = optimized_similarity_for_constrained_memory(
                term_matrix,
                cand_matrix,
                batch_size=adaptive_batch_size_calculator(term_matrix, cand_matrix),
                precision=precision,
            )
        elif is_large_matrix:
            # For moderately large matrices
            logger.debug("Using memory-efficient similarity for large matrix")
            similarity_matrix = memory_efficient_similarity(
                term_matrix,
                cand_matrix,
                batch_size=adaptive_batch_size_calculator(term_matrix, cand_matrix),
            )
        else:
            # For small matrices, use the standard approach
            # Normalize the vectors
            term_norms = np.linalg.norm(term_matrix, axis=1, keepdims=True)
            cand_norms = np.linalg.norm(cand_matrix, axis=1, keepdims=True)

            # Replace zero norms with 1 to avoid division by zero
            term_norms[term_norms == 0] = 1
            cand_norms[term_norms == 0] = 1

            # Normalize
            term_matrix = term_matrix / term_norms
            cand_matrix = cand_matrix / cand_norms

            # Calculate similarity matrix in one operation
            similarity_matrix = np.dot(term_matrix, cand_matrix.T)

        # Ensure values are in [0, 1] range
        similarity_matrix = np.clip(similarity_matrix, 0, 1)

        return similarity_matrix

    def calculate_cross_model_similarity(
        self, term_vector: "VectorType", vector: "VectorType"
    ) -> float:
        """
        Calculate similarity between vectors that might come from different models
        with potentially different dimensions.

        Args:
            term_vector: First vector
            vector: Second vector, potentially with different dimensions

        Returns:
            float: Cosine similarity, properly handled for dimension mismatches
        """
        # Convert to numpy arrays if they're torch tensors
        if torch.is_tensor(term_vector):
            term_vector = term_vector.cpu().numpy()
        if torch.is_tensor(vector):
            vector = vector.cpu().numpy()

        # Handle dimension mismatch
        if len(term_vector) != len(vector):
            # Use zero padding to make vectors the same length
            if len(term_vector) < len(vector):
                # Pad the first vector with zeros
                padded_term_vector = np.zeros(len(vector), dtype=term_vector.dtype)
                padded_term_vector[: len(term_vector)] = term_vector
                term_vector = padded_term_vector
            else:
                # Pad the second vector with zeros
                padded_vector = np.zeros(len(term_vector), dtype=vector.dtype)
                padded_vector[: len(vector)] = vector
                vector = padded_vector

        # Calculate cosine similarity with proper handling for zero vectors
        norm_term = np.linalg.norm(term_vector)
        norm_vector = np.linalg.norm(vector)

        if norm_term > 0 and norm_vector > 0:
            sim = np.dot(term_vector, vector) / (norm_term * norm_vector)
            return float(max(0.0, min(1.0, sim)))  # Ensure result is between 0 and 1
        return 0.0  # Return zero similarity for zero vectors

    def calculate_cross_model_similarity_batch(
        self, term_vector: "VectorType", vectors: Dict[str, "VectorType"]
    ) -> Dict[str, float]:
        """
        Calculate similarity between one vector and multiple vectors that
        might come from different models with potentially different dimensions.

        Args:
            term_vector: Base vector to compare against
            vectors: Dictionary mapping terms to their vectors

        Returns:
            Dict[str, float]: Dictionary mapping terms to similarity scores
        """
        similarities = {}

        # Convert term_vector to numpy if it's a torch tensor
        if torch.is_tensor(term_vector):
            term_vector_np = term_vector.cpu().numpy()
        else:
            term_vector_np = term_vector

        for term, vector in vectors.items():
            similarities[term] = self.calculate_cross_model_similarity(
                term_vector_np, vector
            )

        return similarities

    def _get_vector_dimension(self) -> int:
        """Get the vector dimensionality from the model."""
        # First check if NLP model is set
        if self.nlp is None:
            logger.debug("NLP model not set yet, using default vector dimension")
            return 300  # Default fallback if NLP model not set yet

        # Try to determine vector dimension from model
        try:
            if hasattr(self.nlp, "vector_size"):
                return self.nlp.vector_size

            # Try using dummy document
            doc = self.nlp("test")
            if hasattr(doc, "vector"):
                return len(doc.vector)
        except Exception as e:
            logger.warning(f"Could not determine vector dimension: {e}")

        # Default value if detection fails
        return 300

    def _calculate_and_cache_vector(self, term: str) -> Optional[VectorType]:
        """Calculate vector and cache it with minimal lock scope."""
        # This is a potentially expensive operation that should minimize lock time

        # First, calculate the vector without holding lock
        doc = self.nlp(term)
        if not doc.has_vector:
            return None

        vec = doc.vector

        # Now we need to store in cache which requires minimal lock time
        cache_key = self.cache_manager._create_key(term)

        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(
                self._lock, name=f"vector_cache_{self.namespace}_store"
            ):
                # Store raw numpy array in list format for better compatibility
                self.cache_manager.backend.set(cache_key, vec.tolist())
                self.stats["vector_calcs"] += 1
        else:
            with self._lock:
                # Store raw numpy array in list format for better compatibility
                self.cache_manager.backend.set(cache_key, vec.tolist())
                self.stats["vector_calcs"] += 1

        # Return as torch tensor if GPU is available - no lock needed
        if self.gpu_available:
            try:
                return torch.tensor(vec, device="cuda")
            except Exception as e:
                logger.debug(f"Failed to create GPU tensor: {e}, falling back to numpy")

        return vec

    @contextmanager
    def memory_optimized_operation(self, operation_name: str):
        """Context manager for memory-intensive operations with error handling and monitoring."""
        start_time = time.time()

        try:
            # Track initial memory usage
            initial_mem = psutil.virtual_memory().available / (1024**2)
            logger.debug(
                f"Starting {operation_name} with {initial_mem:.1f}MB available"
            )

            # Check memory before operation
            self._check_and_adjust_memory()
            yield
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"{operation_name}: CUDA out of memory, optimizing...")
                optimize_gpu_memory()
                # Let the caller handle retry logic
                raise
        except Exception as e:
            logger.error(f"Error in {operation_name}: {e}")
            raise
        finally:
            # Calculate operation duration
            duration = time.time() - start_time

            # For longer operations, check memory impact and clean up if needed
            if duration > 1.0:  # Only for non-trivial operations
                try:
                    final_mem = psutil.virtual_memory().available / (1024**2)
                    memory_diff = initial_mem - final_mem

                    logger.debug(
                        f"Completed {operation_name} in {duration:.2f}s "
                        f"(memory impact: {memory_diff:.1f}MB)"
                    )

                    # If significant memory was used, help clean up
                    if memory_diff > 500:  # More than 500MB consumed
                        gc.collect()  # Run garbage collection

                        # Clean GPU memory if operation was large and GPU is used
                        if self.gpu_available:
                            optimize_gpu_memory()
                except Exception as e:
                    logger.debug(f"Error during memory cleanup: {e}")

            # Always check memory after intensive operations
            self._check_and_adjust_memory()

    def adaptive_vector_batch_size(self, vector_dim: int = None) -> int:
        """
        Calculate an optimal batch size for vector operations based on
        available memory and vector dimensions.

        Args:
            vector_dim: Dimension of the vectors (defaults to self.vector_dim)

        Returns:
            int: Recommended batch size
        """
        # Start with configured base size
        base_size = (
            self.config.get("caching", {}).get("vector_cache", {}).get("batch_size", 50)
        )

        if vector_dim is None:
            vector_dim = self.vector_dim

        try:
            # Get available memory in MB
            mem_available = psutil.virtual_memory().available / (1024**2)
            mem_percent = psutil.virtual_memory().percent

            # Estimate bytes per vector (float32 = 4 bytes per element)
            bytes_per_vector = vector_dim * 4

            # Target using at most 50% of available memory for vector data
            target_memory = mem_available * 0.5

            # Calculate how many vectors would fit in target memory
            # We account for overhead by multiplying the vector size by 2.5
            # (original vector + normalized vector + intermediate calculations)
            max_vectors = int(target_memory * 1024 * 1024 / (bytes_per_vector * 2.5))

            # Apply reasonable bounds
            batch_size = max(10, min(max_vectors, 500))

            # Apply memory pressure adjustments from current implementation
            if mem_percent > 80:
                batch_size = max(5, batch_size // 4)
            elif mem_percent > 70:
                batch_size = max(10, batch_size // 2)
            # Increase batch size when memory is plentiful and GPU is available
            elif mem_percent < 40 and self.gpu_available:
                batch_size = min(200, batch_size * 2)

            logger.debug(
                f"Adaptive batch size: {batch_size} (mem: {mem_available:.1f}MB, dim: {vector_dim})"
            )
            return batch_size

        except Exception as e:
            logger.warning(f"Error calculating adaptive batch size: {e}")
            # Default fallback
            return base_size

    def _check_and_adjust_memory(self) -> None:
        """Monitor and optimize memory usage with smarter GC triggers."""
        component_id = f"vector_cache_{self.namespace}_{id(self)}"

        def trim_callback(memory_usage: float) -> None:
            # More aggressive trimming for vector cache as vectors consume more memory
            trim_percent = min(
                50.0, max(30.0, memory_usage - self._max_memory_percent + 10.0)
            )

            trimmed = self.cache_manager.trim(percent=trim_percent)

            if trimmed:
                logger.info(
                    f"Trimmed {trimmed} vector entries ({trim_percent:.1f}%) to reduce memory pressure"
                )

            # More aggressive garbage collection strategy
            if trimmed > 50 or memory_usage > 90:
                logger.info("Running full garbage collection cycle")
                gc.collect(generation=2)  # Force full collection
            elif trimmed > 10:
                gc.collect(generation=1)  # Collect middle generation
            elif trimmed > 0:
                gc.collect(generation=0)  # Quick collection of youngest objects

            # Try to free GPU memory if using it and memory pressure is high
            if self.gpu_available and memory_usage > self._max_memory_percent - 5:
                optimize_gpu_memory()

        # Try using memory_utils.check_memory_usage if available
        try:
            from memory_utils import check_memory_usage as memory_utils_check

            self._last_memory_check = memory_utils_check(
                component_id,
                self._last_memory_check,
                self._memory_check_interval,
                self._max_memory_percent,
                trim_callback,
            )
        except (ImportError, TypeError):
            # Fall back to basic implementation or handle different signature
            updated_time = check_memory_usage(
                component_id, self._last_memory_check, self._memory_check_interval
            )

            # Only proceed with check if enough time has passed
            if updated_time > self._last_memory_check:
                self._last_memory_check = updated_time

                try:
                    memory_usage = psutil.virtual_memory().percent
                    if memory_usage > self._max_memory_percent:
                        trim_callback(memory_usage)
                except Exception as e:
                    logger.warning(f"Memory optimization error: {e}")

    def calculate_similarity_batch(
        self, term: str, candidates: List[str]
    ) -> "SimilarityDict":
        """
        Calculate similarity between a term and multiple candidates efficiently.
        Thread-safe implementation.
        """
        if not candidates:
            return {}

        self.stats["similarity_calcs"] += 1

        # Get vector for the main term
        term_vector = self.get_vector(term)
        if term_vector is None:
            return {candidate: 0.0 for candidate in candidates}

        # Get vectors for all candidates in one batch operation
        candidate_vectors = self.get_vectors_batch(candidates)

        # Calculate similarities - no lock needed for pure computation
        similarities = {}
        for candidate, vector in candidate_vectors.items():
            if vector is not None:
                # Calculate cosine similarity
                norm_term = np.linalg.norm(term_vector)
                norm_cand = np.linalg.norm(vector)

                if norm_term > 0 and norm_cand > 0:
                    sim = np.dot(term_vector, vector) / (norm_term * norm_cand)
                    similarities[candidate] = max(0.0, min(1.0, sim))  # Clamp to [0,1]
                else:
                    similarities[candidate] = 0.0
            else:
                similarities[candidate] = 0.0

        return similarities


class TextProcessingCacheManager:
    """
    Centralized cache manager for text processing operations.
    Handles preprocessing, tokenization, and n-gram generation caches.
    """

    def __init__(self, config: Dict[str, Any], namespace: str = "text_processing"):
        self.config = config
        self.namespace = namespace
        # Calculate optimal cache size based on available memory
        cache_size = calculate_optimal_cache_size(config, "text")

        # Update cache sizes with the calculated value
        self.preprocess_cache = CacheManager(config, f"{namespace}_preprocess")
        self.preprocess_cache.backend.adjust_capacity(cache_size)

        self.tokenize_cache = CacheManager(config, f"{namespace}_tokenize")
        self.tokenize_cache.backend.adjust_capacity(cache_size)

        self.ngram_cache = CacheManager(config, f"{namespace}_ngram")
        self.ngram_cache.backend.adjust_capacity(
            int(cache_size * 0.8)
        )  # N-grams may use more memory

        # Add sentence cache
        self.sentence_cache = CacheManager(config, f"{namespace}_sentence")
        self.sentence_cache.backend.adjust_capacity(int(cache_size * 0.5))

        # Store config hash for cache invalidation
        self.config_hash = hash_config_sections(
            config,
            get_cache_salt(config),
            ["stop_words", "stop_words_add", "stop_words_exclude", "text_processing"],
        )

        # Add lock for thread safety with proper hierarchy
        if LOCK_UTILS_AVAILABLE:
            self._lock = create_component_lock(f"text_cache_{namespace}")
        else:
            self._lock = threading.RLock()

        # Statistics tracking
        self.stats = {
            "preprocess_hits": 0,
            "preprocess_misses": 0,
            "tokenize_hits": 0,
            "tokenize_misses": 0,
            "ngram_hits": 0,
            "ngram_misses": 0,
            "sentence_hits": 0,
            "sentence_misses": 0,
        }

    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate a hash from relevant config sections for cache invalidation."""
        relevant_sections = [
            "stop_words",
            "stop_words_add",
            "stop_words_exclude",
            "text_processing",
            "validation",
        ]

        relevant_config = {}
        for section in relevant_sections:
            if section in config:
                relevant_config[section] = config.get(section)

        config_str = json.dumps(relevant_config, sort_keys=True)
        return xxhash.xxh3_64(config_str.encode()).hexdigest()

    def get_preprocess_cache(self, text: str) -> Optional[str]:
        """Get preprocessed text from cache."""
        with self._lock:
            cache_key = f"preproc:{self.config_hash}:{xxhash.xxh3_64(text.encode()).hexdigest()}"
            result = self.preprocess_cache.get(cache_key)
            if result is not None:
                self.stats["preprocess_hits"] += 1
            else:
                self.stats["preprocess_misses"] += 1
            return result

    def set_preprocess_cache(self, text: str, preprocessed: str) -> None:
        """Store preprocessed text in cache."""
        with self._lock:
            cache_key = f"preproc:{self.config_hash}:{xxhash.xxh3_64(text.encode()).hexdigest()}"
            self.preprocess_cache.set(cache_key, preprocessed)

    def get_tokens_cache(self, text: str) -> Optional[List[str]]:
        """Get tokenized text from cache."""
        with self._lock:
            cache_key = (
                f"tokens:{self.config_hash}:{xxhash.xxh3_64(text.encode()).hexdigest()}"
            )
            result = self.tokenize_cache.get(cache_key)
            if result is not None:
                self.stats["tokenize_hits"] += 1
            else:
                self.stats["tokenize_misses"] += 1
            return result

    def set_tokens_cache(self, text: str, tokens: List[str]) -> None:
        """Store tokenized text in cache."""
        with self._lock:
            cache_key = (
                f"tokens:{self.config_hash}:{xxhash.xxh3_64(text.encode()).hexdigest()}"
            )
            self.tokenize_cache.set(cache_key, tokens)

    def get_ngram_cache(self, tokens_key: str, n: int) -> Optional[Set[str]]:
        """Get n-grams from cache."""
        with self._lock:
            cache_key = f"ngram:{self.config_hash}:{n}:{tokens_key}"
            result = self.ngram_cache.get(cache_key)
            if result is not None:
                self.stats["ngram_hits"] += 1
            else:
                self.stats["ngram_misses"] += 1
            return result

    def set_ngram_cache(self, tokens_key: str, n: int, ngrams: Set[str]) -> None:
        """Store n-grams in cache."""
        with self._lock:
            cache_key = f"ngram:{self.config_hash}:{n}:{tokens_key}"
            self.ngram_cache.set(cache_key, ngrams)

    def clear_all_caches(self) -> None:
        """Clear all text processing caches."""
        with self._lock:
            self.preprocess_cache.clear()
            self.tokenize_cache.clear()
            self.ngram_cache.clear()
            logger.info(f"All {self.namespace} caches cleared")

    def get_stats(self) -> "CacheStats":
        """Get combined stats for all text processing caches."""
        combined_stats = {**self.stats}

        # Add hit ratios
        for cache_type in ["preprocess", "tokenize", "ngram"]:
            hits = combined_stats.get(f"{cache_type}_hits", 0)
            misses = combined_stats.get(f"{cache_type}_misses", 0)
            total = hits + misses
            hit_rate = (hits / total) * 100 if total > 0 else 0
            combined_stats[f"{cache_type}_hit_rate"] = hit_rate

        # Add individual cache stats
        combined_stats["preprocess_stats"] = self.preprocess_cache.get_stats()
        combined_stats["tokenize_stats"] = self.tokenize_cache.get_stats()
        combined_stats["ngram_stats"] = self.ngram_cache.get_stats()

        return cast("CacheStats", combined_stats)


class IntegratedCacheSystem:
    """
    Integrated cache system that coordinates between specialized cache managers.
    Provides a unified interface while leveraging specialized implementations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config

        # Create base cache manager
        self.base_cache: CacheManager = CacheManager(config, "integrated_cache")

        # Initialize specialized cache managers
        self.vector_cache: Optional[VectorCacheManager] = (
            None  # Lazy initialization with NLP model
        )
        self.text_cache: TextProcessingCacheManager = TextProcessingCacheManager(config)

        # Cache statistics - use Dict[str, Any] since we store both ints and floats
        self.stats: Dict[str, Any] = {
            "vector_hits": 0,
            "vector_misses": 0,
            "text_hits": 0,
            "text_misses": 0,
            "general_hits": 0,
            "general_misses": 0,
        }

        # Thread safety - using registered locks with hierarchy
        if LOCK_UTILS_AVAILABLE:
            self._lock = create_component_lock("integrated_cache")
        else:
            self._lock = threading.RLock()

        logger.info("Initialized integrated cache system with specialized managers")

        # Initialize memory manager
        self._memory_manager = None
        try:
            from memory_manager import create_memory_manager

            self._memory_manager = create_memory_manager(self.config)

            # Register initial components
            self._register_with_memory_manager(self.base_cache.backend, "medium")

            # Register text cache components
            self._register_with_memory_manager(
                self.text_cache.preprocess_cache.backend, "medium"
            )
            self._register_with_memory_manager(
                self.text_cache.tokenize_cache.backend, "medium"
            )
            self._register_with_memory_manager(
                self.text_cache.ngram_cache.backend, "medium"
            )
            if hasattr(self.text_cache, "sentence_cache"):
                self._register_with_memory_manager(
                    self.text_cache.sentence_cache.backend, "low"
                )

            logger.info("Memory manager initialized and components registered")
        except (ImportError, Exception) as e:
            if isinstance(e, ImportError):
                logger.warning(
                    "Memory manager module not available, falling back to basic memory management"
                )
            else:
                logger.warning(
                    f"Failed to initialize memory manager: {e}, falling back to basic memory management"
                )
            self._memory_manager = None

    def _register_with_memory_manager(
        self, component: Any, priority: str = "medium"
    ) -> bool:
        """Safely register a component with memory manager if available."""
        if not self._memory_manager or not component:
            return False

        try:
            self._memory_manager.register_component(component, priority)
            return True
        except Exception as e:
            logger.warning(f"Failed to register component with memory manager: {e}")
            return False

    def set_nlp_model(self, nlp) -> None:
        """Set the NLP model for vector cache operations."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(self._lock, name="integrated_cache_set_nlp"):
                if self.vector_cache is None:
                    self.vector_cache = VectorCacheManager(self.config, nlp)
                    # Register with memory manager after creation
                    self._register_with_memory_manager(
                        self.vector_cache.cache_manager.backend, "high"
                    )
                else:
                    self.vector_cache.nlp = nlp
                logger.info("NLP model set for vector cache operations")
        else:
            with self._lock:
                if self.vector_cache is None:
                    self.vector_cache = VectorCacheManager(self.config, nlp)
                    # Register with memory manager after creation
                    self._register_with_memory_manager(
                        self.vector_cache.cache_manager.backend, "high"
                    )
                else:
                    self.vector_cache.nlp = nlp
                logger.info("NLP model set for vector cache operations")

    def get_vector(self, term: str) -> Optional["VectorType"]:
        """Get vector for a term with caching - avoiding nested locks."""
        # First check if vector cache is initialized without holding a lock
        if self.vector_cache is None:
            logger.warning("Vector cache used before NLP model was set")
            return None

        # Store a reference to vector_cache to avoid holding lock during call
        vector_cache_ref = self.vector_cache

        # Get the vector without holding our own lock
        result = vector_cache_ref.get_vector(term)

        # Update statistics under lock - minimized critical section
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(self._lock, name="integrated_cache_stats"):
                if result is not None:
                    self.stats["vector_hits"] += 1
                else:
                    self.stats["vector_misses"] += 1
        else:
            with self._lock:
                if result is not None:
                    self.stats["vector_hits"] += 1
                else:
                    self.stats["vector_misses"] += 1

        return result

    def get_vectors_batch(self, terms: List[str], batch_size: int = 50) -> "VectorDict":
        """Get vectors for multiple terms with batched processing - avoiding nested locks."""
        # Check vector_cache without lock
        if self.vector_cache is None:
            logger.warning("Vector cache used before NLP model was set")
            return {}

        # Store a reference to avoid holding lock during potentially long operation
        vector_cache_ref = self.vector_cache

        # Execute potentially long-running batch operation without holding lock
        result = vector_cache_ref.get_vectors_batch(terms, batch_size)

        # Update statistics with minimal lock time
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(self._lock, name="integrated_cache_batch_stats"):
                hits = sum(1 for term in terms if term in result)
                misses = len(terms) - hits
                self.stats["vector_hits"] += hits
                self.stats["vector_misses"] += misses
        else:
            with self._lock:
                hits = sum(1 for term in terms if term in result)
                misses = len(terms) - hits
                self.stats["vector_hits"] += hits
                self.stats["vector_misses"] += misses

        return result

    def calculate_similarity_batch(
        self, term: str, candidates: List[str]
    ) -> "SimilarityDict":
        """Calculate similarity between term and candidates."""
        # Check without holding lock
        if self.vector_cache is None:
            logger.warning("Vector cache used before NLP model was set")
            return {}

        # Store reference to avoid holding lock during operation
        vector_cache_ref = self.vector_cache

        # Execute potentially long-running operation without lock
        return vector_cache_ref.calculate_similarity_batch(term, candidates)

    def get_preprocessed_text(self, text: str) -> Optional[str]:
        """Get preprocessed text from cache."""
        if LOCK_UTILS_AVAILABLE:
            # Get the result first without holding our lock
            result = self.text_cache.get_preprocess_cache(text)

            # Update stats with minimal lock scope
            with coordinated_lock(self._lock, name="integrated_cache_preprocess_stats"):
                if result is not None:
                    self.stats["text_hits"] += 1
                else:
                    self.stats["text_misses"] += 1
            return result
        else:
            with self._lock:
                # Get the result within the lock
                result = self.text_cache.get_preprocess_cache(text)
                # Update stats within the same lock operation
                if result is not None:
                    self.stats["text_hits"] += 1
                else:
                    self.stats["text_misses"] += 1
                return result

    def set_preprocessed_text(self, text: str, preprocessed: str) -> None:
        """Store preprocessed text in cache."""
        self.text_cache.set_preprocess_cache(text, preprocessed)

    def get_tokens(self, text: str) -> Optional[List[str]]:
        """Get tokenized text from cache."""
        return self.text_cache.get_tokens_cache(text)

    def set_tokens(self, text: str, tokens: List[str]) -> None:
        """Store tokenized text in cache."""
        self.text_cache.set_tokens_cache(text, tokens)

    def get_ngrams(self, tokens_key: str, n: int) -> Optional[Set[str]]:
        """Get n-grams from cache."""
        return self.text_cache.get_ngram_cache(tokens_key, n)

    def set_ngrams(self, tokens_key: str, n: int, ngrams: Set[str]) -> None:
        """Store n-grams in cache."""
        self.text_cache.set_ngram_cache(tokens_key, n, ngrams)

    def get_general(self, key: str, namespace: str = "general") -> Any:
        """Get item from general cache."""
        if LOCK_UTILS_AVAILABLE:
            # Get the result first without holding our lock
            result = self.base_cache.get(f"{namespace}:{key}")

            # Update stats with minimal lock scope
            with coordinated_lock(self._lock, name="integrated_cache_general_stats"):
                if result is not None:
                    self.stats["general_hits"] += 1
                else:
                    self.stats["general_misses"] += 1
            return result
        else:
            with self._lock:
                # Get the result within the lock
                result = self.base_cache.get(f"{namespace}:{key}")
                # Update stats within the same lock operation
                if result is not None:
                    self.stats["general_hits"] += 1
                else:
                    self.stats["general_misses"] += 1
                return result

    def set_general(self, key: str, value: Any, namespace: str = "general") -> None:
        """Set item in general cache."""
        self.base_cache.set(f"{namespace}:{key}", value)

    def get_cache_stats(self) -> "CacheStats":
        """Get combined cache statistics."""
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(self._lock, name="integrated_cache_stats"):
                combined_stats = {**self.stats}

                # Calculate hit rates
                for cache_type in ["vector", "text", "general"]:
                    hits = combined_stats.get(f"{cache_type}_hits", 0)
                    misses = combined_stats.get(f"{cache_type}_misses", 0)
                    total = hits + misses
                    hit_rate = (hits / total) * 100 if total > 0 else 0
                    combined_stats[f"{cache_type}_hit_rate"] = hit_rate

                # Add specialized cache stats if available
                if self.vector_cache is not None:
                    combined_stats["vector_detailed"] = self.vector_cache.get_stats()

                combined_stats["text_detailed"] = self.text_cache.get_stats()
                combined_stats["general_detailed"] = self.base_cache.get_stats()

                return cast("CacheStats", combined_stats)
        else:
            with self._lock:
                combined_stats = {**self.stats}

                # Calculate hit rates
                for cache_type in ["vector", "text", "general"]:
                    hits = combined_stats.get(f"{cache_type}_hits", 0)
                    misses = combined_stats.get(f"{cache_type}_misses", 0)
                    total = hits + misses
                    hit_rate = (hits / total) * 100 if total > 0 else 0
                    combined_stats[f"{cache_type}_hit_rate"] = hit_rate

                # Add specialized cache stats if available
                if self.vector_cache is not None:
                    combined_stats["vector_detailed"] = self.vector_cache.get_stats()

                combined_stats["text_detailed"] = self.text_cache.get_stats()
                combined_stats["general_detailed"] = self.base_cache.get_stats()

                return cast("CacheStats", combined_stats)

    def clear_caches(self, cache_types: Optional[List[str]] = None) -> None:
        """
        Clear selected or all caches.

        Args:
            cache_types: List of cache types to clear. If None, clears all caches.
                         Valid options: "vector", "text", "general", "all"
        """
        if LOCK_UTILS_AVAILABLE:
            with coordinated_lock(self._lock, name="integrated_cache_clear"):
                if cache_types is None:
                    cache_types = ["all"]

                if "all" in cache_types or "general" in cache_types:
                    self.base_cache.clear()
                    logger.info("General cache cleared")

                if "all" in cache_types or "text" in cache_types:
                    self.text_cache.clear_all_caches()
                    logger.info("Text processing caches cleared")

                if (
                    "all" in cache_types or "vector" in cache_types
                ) and self.vector_cache:
                    self.vector_cache.cache_manager.clear()
                    logger.info("Vector cache cleared")

                if "all" in cache_types:
                    # Reset statistics
                    self.stats = {key: 0 for key in self.stats}
                    logger.info("All caches cleared and statistics reset")
        else:
            with self._lock:
                if cache_types is None:
                    cache_types = ["all"]

                if "all" in cache_types or "general" in cache_types:
                    self.base_cache.clear()
                    logger.info("General cache cleared")

                if "all" in cache_types or "text" in cache_types:
                    self.text_cache.clear_all_caches()
                    logger.info("Text processing caches cleared")

                if (
                    "all" in cache_types or "vector" in cache_types
                ) and self.vector_cache:
                    self.vector_cache.cache_manager.clear()
                    logger.info("Vector cache cleared")

                if "all" in cache_types:
                    # Reset statistics
                    self.stats = {key: 0 for key in self.stats}
                    logger.info("All caches cleared and statistics reset")

    def optimize_memory_usage(self) -> "MemoryOptimizationResult":
        """
        Optimize memory usage across all caches with coordination.
        Uses MemoryManager when available, with fallback to direct trimming.

        Returns:
            Dictionary with number of items trimmed per cache type.
        """
        result: Dict[str, int] = {"general": 0, "text": 0, "vector": 0}

        # Try to use MemoryManager for coordinated optimization
        if self._memory_manager:
            try:
                # Capture sizes before optimization for better result reporting
                sizes_before = self._get_cache_sizes()

                # Let memory manager handle the optimization
                optimization_performed = self._memory_manager.optimize_memory(
                    level="standard"
                )

                if optimization_performed:
                    # Calculate how many items were trimmed based on before/after comparison
                    sizes_after = self._get_cache_sizes()
                    result["general"] = max(
                        0, sizes_before["general"] - sizes_after["general"]
                    )
                    result["text"] = max(0, sizes_before["text"] - sizes_after["text"])
                    result["vector"] = max(
                        0, sizes_before["vector"] - sizes_after["vector"]
                    )

                    # Log results
                    total_trimmed = sum(result.values())
                    if total_trimmed > 0:
                        # Report memory usage after optimization
                        try:
                            memory_status = self._memory_manager.get_memory_status()
                            if "percent" in memory_status:
                                logger.info(
                                    f"Memory optimization: trimmed {total_trimmed} items, "
                                    f"memory usage: {memory_status['percent']:.1f}%"
                                )
                        except Exception as e:
                            logger.debug(f"Error getting memory status: {e}")

                return result

            except Exception as e:
                logger.warning(
                    f"Memory manager optimization failed: {e}, falling back to direct trimming"
                )
                # Fall through to direct trimming

        # Fallback: direct trimming
        logger.debug("Using direct memory optimization")

        # Check current memory usage - no lock needed
        try:
            memory_usage = psutil.virtual_memory().percent
            # Use thresholds from configuration
            emergency_threshold = self.config.get("hardware_limits", {}).get(
                "emergency_percent", 85
            )
            critical_threshold = self.config.get("hardware_limits", {}).get(
                "critical_percent", 75
            )
            warning_threshold = self.config.get("hardware_limits", {}).get(
                "warning_percent", 65
            )

            # Set trim percentages based on memory pressure
            if memory_usage > emergency_threshold:  # Emergency
                trim_percents = {
                    "general": 30,
                    "text": 40,
                    "vector": 20,
                }  # Aggressive trim
            elif memory_usage > critical_threshold:  # Critical
                trim_percents = {"general": 20, "text": 25, "vector": 15}  # Higher trim
            elif memory_usage > warning_threshold:  # Warning
                trim_percents = {
                    "general": 10,
                    "text": 15,
                    "vector": 8,
                }  # Standard trim
            else:
                trim_percents = {"general": 5, "text": 8, "vector": 3}  # Light trim
        except Exception:
            # Default if memory check fails
            trim_percents = {"general": 10, "text": 15, "vector": 8}  # Default trim

        # Trim general cache with proper error handling
        try:
            result["general"] = self.base_cache.trim(trim_percents["general"])
        except Exception as e:
            logger.error(f"Error trimming general cache: {e}")

        # Trim text caches with proper error handling
        try:
            text_result = 0
            text_result += self.text_cache.preprocess_cache.trim(trim_percents["text"])
            text_result += self.text_cache.tokenize_cache.trim(trim_percents["text"])
            text_result += self.text_cache.ngram_cache.trim(trim_percents["text"])
            if hasattr(self.text_cache, "sentence_cache"):
                text_result += self.text_cache.sentence_cache.trim(
                    trim_percents["text"]
                )
            result["text"] = text_result
        except Exception as e:
            logger.error(f"Error trimming text caches: {e}")

        # Trim vector cache if available
        if self.vector_cache:
            try:
                result["vector"] = self.vector_cache.cache_manager.trim(
                    trim_percents["vector"]
                )
                # Also optimize GPU memory if using vectors
                if (
                    hasattr(self.vector_cache, "gpu_available")
                    and self.vector_cache.gpu_available
                ):
                    optimize_gpu_memory()
            except Exception as e:
                logger.error(f"Error trimming vector cache: {e}")

        # Log results and run GC if needed
        total_trimmed = sum(result.values())
        if total_trimmed > 0:
            logger.info(f"Memory optimization: trimmed {total_trimmed} cache entries")
            # Use tiered garbage collection based on trim size
            if total_trimmed > 2000:
                gc.collect(2)  # Full collection for large trims
                logger.debug("Performed full garbage collection")
            elif total_trimmed > 500:
                gc.collect(1)  # Collect generations 0 and 1 for medium trims
                logger.debug("Performed standard garbage collection")
            elif total_trimmed > 100:
                gc.collect(0)  # Only collect youngest generation for small trims
                logger.debug("Performed fast garbage collection")

        return result

    def _get_cache_sizes(self) -> Dict[str, int]:
        """Get the current size of all cache components."""
        sizes = {"general": 0, "text": 0, "vector": 0}

        # Get general cache size
        try:
            if hasattr(self.base_cache, "backend") and hasattr(
                self.base_cache.backend, "memory_cache"
            ):
                sizes["general"] = len(self.base_cache.backend.memory_cache)
        except Exception:
            pass

        # Get text cache sizes
        try:
            sizes["text"] = 0
            if hasattr(self.text_cache, "preprocess_cache") and hasattr(
                self.text_cache.preprocess_cache.backend, "memory_cache"
            ):
                sizes["text"] += len(
                    self.text_cache.preprocess_cache.backend.memory_cache
                )
            if hasattr(self.text_cache, "tokenize_cache") and hasattr(
                self.text_cache.tokenize_cache.backend, "memory_cache"
            ):
                sizes["text"] += len(
                    self.text_cache.tokenize_cache.backend.memory_cache
                )
            if hasattr(self.text_cache, "ngram_cache") and hasattr(
                self.text_cache.ngram_cache.backend, "memory_cache"
            ):
                sizes["text"] += len(self.text_cache.ngram_cache.backend.memory_cache)
            if hasattr(self.text_cache, "sentence_cache") and hasattr(
                self.text_cache.sentence_cache.backend, "memory_cache"
            ):
                sizes["text"] += len(
                    self.text_cache.sentence_cache.backend.memory_cache
                )
        except Exception:
            pass

        # Get vector cache size
        try:
            if (
                self.vector_cache
                and hasattr(self.vector_cache, "cache_manager")
                and hasattr(self.vector_cache.cache_manager.backend, "memory_cache")
            ):
                sizes["vector"] = len(
                    self.vector_cache.cache_manager.backend.memory_cache
                )
        except Exception:
            pass

        return sizes

    def create_enhanced_cache_manager(
        self, namespace: str = "enhanced_cache"
    ) -> EnhancedCacheManager:
        """
        Create and return an enhanced cache manager with dependency tracking.

        Args:
            namespace: Cache namespace

        Returns:
            EnhancedCacheManager: The newly created cache manager
        """
        cache_manager = EnhancedCacheManager(self.config, namespace)

        # Register with memory manager if available
        self._register_with_memory_manager(cache_manager.backend, "medium")

        return cache_manager

    def get_sentences(self, text: str) -> Optional[List[str]]:
        """Get extracted sentences from cache."""
        cache_key = f"sentences:{xxhash.xxh3_64(text.encode()).hexdigest()}"
        return self.get_general(cache_key, namespace="sentences")

    def set_sentences(self, text: str, sentences: List[str]) -> None:
        """Store extracted sentences in cache."""
        cache_key = f"sentences:{xxhash.xxh3_64(text.encode()).hexdigest()}"
        self.set_general(cache_key, sentences, namespace="sentences")

    def batch_process_with_vectors(
        self, texts: List[str], process_fn: "VectorProcessCallback"
    ) -> Dict[str, Any]:
        """
        Process multiple texts with their vectors in an optimized batch operation.
        This creates synergy between text processing and vector caching.

        Args:
            texts: List of text items to process
            process_fn: Callback function that processes each (text, vector) pair

        Returns:
            Dict mapping texts to their processed results
        """
        results: Dict[str, Any] = {}

        if self.vector_cache is None:
            logger.warning("Vector cache used before NLP model was set")
            return {text: None for text in texts}

        # First get all vectors in one batch operation
        unique_texts = list(set(texts))
        vectors_batch = self.get_vectors_batch(unique_texts)

        # Process each text with its vector
        for text in unique_texts:
            if text in vectors_batch and vectors_batch[text] is not None:
                try:
                    # Call the provided function with text and its vector
                    result = process_fn(text, vectors_batch[text])
                    results[text] = result
                except Exception as e:
                    logger.error(f"Error processing '{text}': {e}")
                    results[text] = None
            else:
                results[text] = None

        return results

    def invalidate_by_config_change(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Perform selective invalidation based on configuration changes,
        coordinated across all cache systems.

        Args:
            old_config: Previous configuration
            new_config: New configuration

        Returns:
            Dict mapping cache types to number of entries invalidated
        """
        invalidation_counts = {"general": 0, "text": 0, "vector": 0, "validation": 0}
        changed_sections = set()

        # First, preserve original behavior: call invalidate_by_config_change on base_cache
        # if it exists, as it may have specialized logic
        if hasattr(self.base_cache, "invalidate_by_config_change"):
            invalidated_sections = self.base_cache.invalidate_by_config_change(
                old_config, new_config
            )
            changed_sections.update(invalidated_sections)

        # Try to use advanced detection if available
        try:
            from cache_invalidation import analyze_config_for_invalidation

            # Get detailed invalidation plan for different cache systems
            invalidation_plan = analyze_config_for_invalidation(old_config, new_config)

            # Handle vector cache invalidation
            if self.vector_cache and (
                "all" in invalidation_plan.get("integrated_cache", [])
                or invalidation_plan.get("vector_cache")
            ):
                try:
                    # Use timestamp-based invalidation if available
                    if hasattr(self.vector_cache, "invalidate_all"):
                        self.vector_cache.invalidate_all()
                        logger.info(
                            "Applied timestamp-based invalidation to vector cache"
                        )
                    else:
                        self.vector_cache.cache_manager.clear()
                        logger.info("Cleared vector cache completely")
                    invalidation_counts["vector"] = 1  # Simplified count
                except Exception as e:
                    logger.error(f"Failed to invalidate vector cache: {e}")

            # Handle text cache invalidation
            if "all" in invalidation_plan.get(
                "integrated_cache", []
            ) or invalidation_plan.get("text_cache"):
                self.text_cache.clear_all_caches()
                invalidation_counts["text"] = 1

            # Handle general cache invalidation
            if "all" in invalidation_plan.get(
                "integrated_cache", []
            ) or invalidation_plan.get("general_cache"):
                self.base_cache.clear()
                invalidation_counts["general"] = 1

        except ImportError:
            # Fall back to basic section comparison if advanced detection isn't available
            logger.info(
                "cache_invalidation module not found, using basic section comparison"
            )

            # If changed_sections is empty (base_cache didn't detect changes), do our own check
            if not changed_sections:
                # Check sections that affect different cache types
                section_checks = {
                    "text": [
                        "text_processing",
                        "stop_words",
                        "stop_words_add",
                        "stop_words_exclude",
                    ],
                    "vector": ["vectorization", "hardware_limits"],
                    "general": ["caching"],
                }

                # Check each section group
                for cache_type, sections in section_checks.items():
                    for section in sections:
                        old_section = old_config.get(section, {})
                        new_section = new_config.get(section, {})

                        if old_section != new_section:
                            # Calculate section-specific hashes for more granular comparison
                            old_hash = xxhash.xxh3_64(
                                json.dumps(old_section, sort_keys=True).encode()
                            ).hexdigest()
                            new_hash = xxhash.xxh3_64(
                                json.dumps(new_section, sort_keys=True).encode()
                            ).hexdigest()

                            if old_hash != new_hash:
                                changed_sections.add(section)

            # Apply invalidation based on section changes
            if changed_sections:
                # Text processing cache
                if any(
                    section in changed_sections
                    for section in [
                        "text_processing",
                        "stop_words",
                        "stop_words_add",
                        "stop_words_exclude",
                    ]
                ):
                    self.text_cache.clear_all_caches()
                    invalidation_counts["text"] = 1  # Simplified count

                # Vector cache
                if any(
                    section in changed_sections
                    for section in ["vectorization", "hardware_limits"]
                ):
                    if self.vector_cache:
                        try:
                            # Use timestamp-based invalidation if available
                            if hasattr(self.vector_cache, "invalidate_all"):
                                self.vector_cache.invalidate_all()
                            else:
                                self.vector_cache.cache_manager.clear()
                            invalidation_counts["vector"] = 1  # Simplified count
                        except Exception as e:
                            logger.error(f"Failed to invalidate vector cache: {e}")

                # General cache - if any caching settings changed
                if "caching" in changed_sections:
                    self.base_cache.clear()
                    invalidation_counts["general"] = 1  # Simplified count

        # Update configuration in all components
        self.config = new_config

        # Update config hash in all cache components that support it
        for cache_component in [
            self.base_cache,
            self.text_cache.preprocess_cache,
            self.text_cache.tokenize_cache,
            self.text_cache.ngram_cache,
        ]:
            if hasattr(cache_component, "update_config_hash"):
                try:
                    cache_component.update_config_hash(new_config)
                except Exception as e:
                    logger.warning(f"Failed to update config hash: {e}")

        # Update vector cache config if it exists
        if self.vector_cache and hasattr(
            self.vector_cache.cache_manager, "update_config_hash"
        ):
            try:
                self.vector_cache.cache_manager.update_config_hash(new_config)
            except Exception as e:
                logger.warning(f"Failed to update vector cache config hash: {e}")

        # Log the invalidation
        if sum(invalidation_counts.values()) > 0:
            invalidated_types = [k for k, v in invalidation_counts.items() if v > 0]
            logger.info(
                f"Cache invalidation applied to: {', '.join(invalidated_types)}"
            )

        return invalidation_counts

    def set_with_dependencies(
        self,
        key: str,
        value: Any,
        namespace: str = "general",
        depends_on: Union[str, List[str], None] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set an item in general cache with dependency tracking.

        Args:
            key: Cache key
            value: Value to store
            namespace: Cache namespace
            depends_on: Key(s) that this entry depends on
            ttl: Optional TTL in seconds

        Returns:
            bool: Success status
        """
        if hasattr(self.base_cache, "set_with_dependencies"):
            return self.base_cache.set_with_dependencies(
                f"{namespace}:{key}", value, depends_on, ttl
            )
        else:
            # Fall back to regular set if enhanced functionality not available
            return self.base_cache.set(f"{namespace}:{key}", value, ttl)

    def invalidate_cache_key(self, key: str, namespace: str = "general") -> int:
        """
        Invalidate a specific cache key and its dependents.

        Args:
            key: The key to invalidate
            namespace: Cache namespace

        Returns:
            int: Number of entries invalidated
        """
        if hasattr(self.base_cache, "invalidate_key"):
            return self.base_cache.invalidate_key(f"{namespace}:{key}")
        else:
            # Fall back to checking and removing the key directly
            cache_key = self.base_cache._create_key(f"{namespace}:{key}")
            if self.base_cache.contains(cache_key):
                # Use clear() as we don't have a direct way to remove a single key
                # This is inefficient but ensures the key is removed
                self.base_cache.clear()
                return 1
            return 0

    def create_enhanced_cache_manager(
        self, namespace: str = "enhanced_cache"
    ) -> EnhancedCacheManager:
        """
        Create and return an enhanced cache manager with dependency tracking.

        Args:
            namespace: Cache namespace

        Returns:
            EnhancedCacheManager: The newly created cache manager
        """
        cache_manager = EnhancedCacheManager(self.config, namespace)

        # Register with memory manager if available
        self._register_with_memory_manager(cache_manager.backend, "medium")

        return cache_manager


# New class added to support SentenceCacheManager export
class SentenceCacheManager(CacheManager):
    """
    Specialized cache manager for sentence caching.
    Inherits from CacheManager.
    """

    pass


class TypedCacheManager(Generic[K, V]):
    """Type-safe cache manager."""

    def __init__(self, config: Dict[str, Any], namespace: str = "general"):
        self.cache_manager = CacheManager(config, namespace)

    def get(self, key: K) -> Optional[V]:
        """Get a strongly-typed item from the cache."""
        return self.cache_manager.get(str(key))

    def set(self, key: K, value: V, ttl: Optional[int] = None) -> bool:
        """Set a strongly-typed item in the cache."""
        return self.cache_manager.set(str(key), value, ttl)

    # Additional methods as needed


# Memory-efficient similarity calculation functions
def memory_efficient_similarity(vec1, vec2, batch_size: Optional[int] = None):
    """
    Calculate cosine similarity with memory efficiency in mind.
    Suitable for large vectors on memory-constrained systems.

    Args:
        vec1: First vector or matrix
        vec2: Second vector or matrix
        batch_size: Optional batch size for large matrix operations

    Returns:
        Similarity score(s)
    """
    import numpy as np

    # Handle vector normalization without creating unnecessary copies
    def normalize_vector(v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v

    # Simple vector-vector case
    if len(vec1.shape) == 1 and len(vec2.shape) == 1:
        vec1_normalized = normalize_vector(vec1)
        vec2_normalized = normalize_vector(vec2)
        return np.dot(vec1_normalized, vec2_normalized)

    # Matrix-matrix case with potential batching
    elif len(vec1.shape) == 2 and len(vec2.shape) == 2:
        # If matrices are small enough, process directly
        if vec1.shape[0] * vec2.shape[0] < 10000 or batch_size is None:
            # Normalize rows
            vec1_norms = np.linalg.norm(vec1, axis=1, keepdims=True)
            vec2_norms = np.linalg.norm(vec2, axis=1, keepdims=True)

            # Replace zero norms with 1
            vec1_norms[vec1_norms == 0] = 1
            vec2_norms[vec2_norms == 0] = 1

            vec1_normalized = vec1 / vec1_norms
            vec2_normalized = vec2 / vec2_norms

            return np.dot(vec1_normalized, vec2_normalized.T)

        # For large matrices, use batching
        else:
            results = np.zeros((vec1.shape[0], vec2.shape[0]))

            # Normalize vec2 once outside the loop - avoid redundant computation
            vec2_norms = np.linalg.norm(vec2, axis=1, keepdims=True)
            vec2_norms[vec2_norms == 0] = 1
            vec2_normalized = vec2 / vec2_norms

            # Process vec1 in batches
            for i in range(0, vec1.shape[0], batch_size):
                end_i = min(i + batch_size, vec1.shape[0])
                vec1_batch = vec1[i:end_i]

                # Normalize this batch
                vec1_norms = np.linalg.norm(vec1_batch, axis=1, keepdims=True)
                vec1_norms[vec1_norms == 0] = 1
                vec1_batch_norm = vec1_batch / vec1_norms

                # Compute similarity for this batch with all of vec2_normalized
                results[i:end_i, :] = np.dot(vec1_batch_norm, vec2_normalized.T)

            return results

    # vector-matrix or matrix-vector case
    else:
        # Ensure vec1 is the matrix if one of them is a vector
        if len(vec1.shape) == 1:
            vec1, vec2 = vec2, vec1
            transpose_result = True
        else:
            transpose_result = False

        # Normalize matrix rows
        vec1_norms = np.linalg.norm(vec1, axis=1, keepdims=True)
        vec1_norms[vec1_norms == 0] = 1
        vec1_normalized = vec1 / vec1_norms

        # Normalize vector
        vec2_norm = np.linalg.norm(vec2)
        if vec2_norm > 0:
            vec2_normalized = vec2 / vec2_norm
        else:
            vec2_normalized = vec2

        result = np.dot(vec1_normalized, vec2_normalized)

        if transpose_result:
            return result.T
        return result


def optimized_similarity_for_constrained_memory(
    vec1, vec2, batch_size: Optional[int] = None, precision: str = "float32"
) -> Union[float, np.ndarray]:
    """
    Calculate cosine similarity with extreme memory efficiency for 8GB systems.
    Uses lower precision and more aggressive batching when needed.

    Args:
        vec1: First vector or matrix
        vec2: Second vector or matrix
        batch_size: Optional batch size for large matrix operations
        precision: Data type precision ("float32" or "float16")

    Returns:
        Similarity score(s)
    """
    import numpy as np

    # Check if we should use lower precision for very large matrices
    if (
        isinstance(vec1, np.ndarray)
        and isinstance(vec2, np.ndarray)
        and len(vec1.shape) == 2
        and len(vec2.shape) == 2
        and vec1.shape[0] * vec2.shape[0] * vec1.shape[1] > 10000000
    ):
        # For very large matrices, use float16 to halve memory requirements
        dtype = np.float16
        logger.debug("Using float16 precision for large similarity matrix")
    else:
        # Otherwise use specified precision (default float32)
        dtype = np.float32 if precision == "float32" else np.float16

    # Simple vector-vector case
    if len(vec1.shape) == 1 and len(vec2.shape) == 1:
        # Convert to specified precision to save memory
        vec1_fp = vec1.astype(dtype)
        vec2_fp = vec2.astype(dtype)

        # Normalize with memory efficiency in mind
        norm1 = np.linalg.norm(vec1_fp)
        norm2 = np.linalg.norm(vec2_fp)

        if norm1 > 0 and norm2 > 0:
            return float(np.dot(vec1_fp / norm1, vec2_fp / norm2))
        return 0.0

    # Matrix case - use the most memory-efficient approach
    elif len(vec1.shape) == 2 and len(vec2.shape) == 2:
        # For large matrices, always use batching
        if batch_size is None:
            # Auto-determine batch size based on dimensions
            vec_dim = vec1.shape[1]
            max_vectors = min(vec1.shape[0], vec2.shape[0])

            # Estimate batch size that uses ~100MB of RAM for intermediate results
            # (conservative for 8GB systems)
            batch_size = max(5, min(100, int(25000000 / (vec_dim * max_vectors))))
            logger.debug(f"Auto-determined batch size: {batch_size}")

        # Process in batches
        results = np.zeros((vec1.shape[0], vec2.shape[0]), dtype=dtype)

        # Normalize vec2 outside the loop to avoid redundant calculations
        vec2_fp = vec2.astype(dtype)
        vec2_norms = np.linalg.norm(vec2_fp, axis=1, keepdims=True)
        vec2_norms[vec2_norms == 0] = 1
        vec2_normalized = vec2_fp / vec2_norms

        for i in range(0, vec1.shape[0], batch_size):
            end_i = min(i + batch_size, vec1.shape[0])
            vec1_batch = vec1[i:end_i].astype(dtype)

            # Normalize this batch
            vec1_norms = np.linalg.norm(vec1_batch, axis=1, keepdims=True)
            vec1_norms[vec1_norms == 0] = 1
            vec1_batch_norm = vec1_batch / vec1_norms

            # Compute similarity for this batch
            results[i:end_i, :] = np.dot(vec1_batch_norm, vec2_normalized.T)

            # Explicitly delete intermediate results to free memory
            del vec1_batch, vec1_norms, vec1_batch_norm

        return results

    # Handle vector-matrix cases
    else:
        # Convert to matrix-vector case if needed
        if len(vec1.shape) == 1:
            return optimized_similarity_for_constrained_memory(
                vec2, vec1.reshape(1, -1), batch_size, precision
            ).T

        # Handle matrix-vector case
        vec1_fp = vec1.astype(dtype)
        vec2_fp = vec2.astype(dtype)

        # Normalize matrix rows
        vec1_norms = np.linalg.norm(vec1_fp, axis=1, keepdims=True)
        vec1_norms[vec1_norms == 0] = 1
        vec1_normalized = vec1_fp / vec1_norms

        # Normalize vector
        vec2_norm = np.linalg.norm(vec2_fp)
        vec2_normalized = vec2_fp / vec2_norm if vec2_norm > 0 else vec2_fp

        # Calculate dot product
        return np.dot(vec1_normalized, vec2_normalized)


# -----------------------------------------------------------------------------
# Explicitly export key classes and functions
# -----------------------------------------------------------------------------

__all__ = [
    # Core classes
    "CacheManager",
    "CacheBackend",
    "MemoryCacheBackend",
    "HybridCacheBackend",
    "EnhancedHybridCacheBackend",
    "VectorCacheManager",
    "TextProcessingCacheManager",
    "IntegratedCacheSystem",
    "SentenceCacheManager",
    "TypedCacheManager",
    "EnhancedCacheManager",
    "AccessCountingMixin",
    "SerializationManager",  # Added new class
    # Utility functions
    "get_cache_salt",
    "calculate_optimal_cache_size",
    "hash_config_sections",
    "check_memory_usage",
    "secure_cache_directory",
    "optimize_gpu_memory",
    # Key helper classes
    "CacheKey",
    "CACHE_VERSION",
]
