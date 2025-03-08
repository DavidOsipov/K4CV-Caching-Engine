"""
Lock utilities for Keywords4CV components.
Provides a coordinated lock management system to prevent deadlocks.
"""

import threading
import time
import logging
import functools
from contextlib import contextmanager
from typing import Dict, Set, Optional, Any, Callable

logger = logging.getLogger(__name__)

# Global lock registry to track lock order and detect potential deadlocks
_lock_registry = {}
_thread_lock_stack = {}
_global_registry_lock = threading.RLock()

# Lock hierarchy levels - higher numbers have higher priority
LOCK_HIERARCHY = {
    "memory_manager": 100,
    "integrated_cache": 90,
    "vector_cache": 80,
    "text_cache": 70,
    "base_cache": 60,
    "backend": 50,
    "component": 40,
}


class LockOrderViolation(Exception):
    """Exception raised when locks are acquired out of order."""

    pass


class LockTimeout(Exception):
    """Exception raised when a lock cannot be acquired within the timeout."""

    pass


def get_hierarchy_level(lock_name: str) -> int:
    """Get the hierarchy level for a lock name."""
    for prefix, level in LOCK_HIERARCHY.items():
        if lock_name.startswith(prefix):
            return level
    return 0  # Default level for unnamed locks


def register_lock(lock_obj: threading.RLock, name: str) -> None:
    """
    Register a lock with the given name in the global registry.

    Args:
        lock_obj: The lock object
        name: A unique name for the lock, used for hierarchy
    """
    with _global_registry_lock:
        _lock_registry[id(lock_obj)] = {
            "name": name,
            "level": get_hierarchy_level(name),
            "object": lock_obj,
        }
        logger.debug(f"Registered lock: {name} with level {get_hierarchy_level(name)}")


def unregister_lock(lock_obj: threading.RLock) -> None:
    """
    Remove a lock from the registry.

    Args:
        lock_obj: The lock object to unregister
    """
    with _global_registry_lock:
        if id(lock_obj) in _lock_registry:
            name = _lock_registry[id(lock_obj)]["name"]
            del _lock_registry[id(lock_obj)]
            logger.debug(f"Unregistered lock: {name}")


@contextmanager
def coordinated_lock(
    lock_obj: threading.RLock, timeout: float = 10.0, name: str = None
) -> None:
    """
    Context manager for acquiring locks in a coordinated manner that prevents deadlocks.

    Args:
        lock_obj: The lock to acquire
        timeout: Maximum time to wait for lock acquisition in seconds
        name: Optional name for unregistered locks

    Raises:
        LockOrderViolation: If acquiring this lock would violate the hierarchy
        LockTimeout: If the lock cannot be acquired within the timeout
    """
    thread_id = threading.get_ident()
    lock_id = id(lock_obj)

    # Check if we already hold this lock (reentrant case)
    thread_locks = _thread_lock_stack.setdefault(thread_id, [])
    if lock_id in [id(l) for l in thread_locks]:
        # Already holding this lock, proceed without re-acquiring
        yield
        return

    # Get lock info from registry
    with _global_registry_lock:
        if lock_id not in _lock_registry and name:
            # Register the lock if it's not already registered
            register_lock(lock_obj, name)

        lock_info = _lock_registry.get(lock_id, {"name": "unnamed", "level": 0})

        # Check for lock order violations based on hierarchy
        for held_lock in thread_locks:
            held_lock_info = _lock_registry.get(
                id(held_lock), {"name": "unknown", "level": 0}
            )
            if held_lock_info["level"] < lock_info["level"]:
                error_msg = (
                    f"Lock order violation: trying to acquire {lock_info['name']} "
                    f"(level {lock_info['level']}) while holding {held_lock_info['name']} "
                    f"(level {held_lock_info['level']})"
                )
                logger.error(error_msg)
                raise LockOrderViolation(error_msg)

    # Try to acquire the lock with timeout
    start_time = time.time()
    while True:
        if lock_obj.acquire(blocking=False):
            try:
                # Track that we now hold this lock
                thread_locks.append(lock_obj)
                yield
            finally:
                # Remove from our tracking when released
                if lock_obj in thread_locks:
                    thread_locks.remove(lock_obj)
                lock_obj.release()
            return

        # Check for timeout
        if time.time() - start_time > timeout:
            error_msg = (
                f"Timeout waiting for lock {lock_info['name']} after {timeout} seconds"
            )
            logger.error(error_msg)
            raise LockTimeout(error_msg)

        # Small sleep to avoid CPU spinning
        time.sleep(0.001)


def synchronized(lock_name: str, timeout: float = 10.0):
    """
    Decorator for synchronizing methods with a named lock.

    Args:
        lock_name: Base name for the lock (will be prefixed with class name)
        timeout: Maximum time to wait for lock acquisition

    Returns:
        Decorator function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create or get lock from object
            if not hasattr(self, "_locks"):
                self._locks = {}

            # Use class name + lock name for better identification
            full_lock_name = f"{self.__class__.__name__}_{lock_name}"

            if full_lock_name not in self._locks:
                lock = threading.RLock()
                self._locks[full_lock_name] = lock
                register_lock(lock, full_lock_name)

            # Use the coordinated lock
            with coordinated_lock(self._locks[full_lock_name], timeout, full_lock_name):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def create_component_lock(component_name: str) -> threading.RLock:
    """
    Create a registered lock for a component.

    Args:
        component_name: Name of the component

    Returns:
        A registered RLock
    """
    lock = threading.RLock()
    register_lock(lock, f"component_{component_name}")
    return lock
