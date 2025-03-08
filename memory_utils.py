"""
Memory utilities for Keywords4CV components.
Provides standardized ways to integrate with the MemoryManager.
"""

import logging
import time
import gc
import psutil
import threading
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, cast, Union, Tuple

logger = logging.getLogger(__name__)

# Global reference to memory manager (will be set by framework initialization)
_memory_manager = None


def set_memory_manager(manager) -> None:
    """Set the global memory manager reference."""
    global _memory_manager
    _memory_manager = manager


def get_memory_manager():
    """Get the global memory manager reference."""
    return _memory_manager


class ComponentMemoryMonitor:
    """
    Helper class for component-level memory monitoring that integrates
    with the centralized MemoryManager.
    """

    def __init__(self, component_name: str, component: Any, priority: str = "medium"):
        """
        Initialize a component memory monitor.

        Args:
            component_name: Name of the component for logging
            component: The component object (must have trim() method)
            priority: Component priority ("high", "medium", "low")
        """
        self.component_name = component_name
        self.component = component
        self.priority = priority
        self.component_id = f"{component_name}_{id(component)}"

        # Check if component supports migration
        self.supports_migration = hasattr(component, "migrate_to_disk") and callable(
            getattr(component, "migrate_to_disk")
        )

        # Track last check to avoid too frequent checks
        self.last_check_time = 0
        self.check_interval = 60  # Default interval in seconds

        # Register with memory manager if available
        self._register_with_memory_manager()

        # Use coordinated lock system if available
        try:
            from lock_utils import create_component_lock

            self._lock = create_component_lock(f"memory_monitor_{component_name}")
        except ImportError:
            self._lock = threading.RLock()

    def _register_with_memory_manager(self) -> None:
        """Register this component with the global memory manager if available."""
        manager = get_memory_manager()
        if manager is not None:
            manager.register_component(self.component, self.priority)
            logger.debug(
                f"Component {self.component_name} registered with memory manager"
            )

    def check_memory(self, force: bool = False) -> bool:
        """
        Check if memory optimization is needed, coordinating with the memory manager.
        Improved to prioritize migration to disk over trimming.
        """
        # Skip check if memory manager exists and says we shouldn't check
        manager = get_memory_manager()
        current_time = time.time()

        # Only check if forced, enough time has passed, or we have no manager
        if not force and (current_time - self.last_check_time < self.check_interval):
            return False

        # Update last check time without lock - atomic operation
        self.last_check_time = current_time

        try:
            memory_usage = psutil.virtual_memory().percent
            optimized = False

            if manager is not None:
                # Ask manager if we should optimize and how much to trim/migrate
                # This call is designed to be safe without holding our lock
                should_optimize, optimization_percent = manager.component_memory_check(
                    self.component_id, memory_usage
                )

                if should_optimize and optimization_percent > 0:
                    # First try migration if component supports it
                    if self.supports_migration:
                        try:
                            # Get a reference to component.migrate_to_disk outside of lock
                            component = self.component
                            migrated = component.migrate_to_disk(optimization_percent)

                            if migrated > 0:
                                logger.info(
                                    f"Component {self.component_name} migrated {migrated} items to disk"
                                )
                                optimized = True
                        except Exception as e:
                            logger.warning(
                                f"Migration error for {self.component_name}: {e}"
                            )

                    # If migration failed or component doesn't support it, try trimming
                    if not optimized:
                        component = self.component
                        if hasattr(component, "trim") and callable(component.trim):
                            # Call trim without holding our lock to prevent deadlocks
                            trimmed = component.trim(optimization_percent)

                            if trimmed > 0:
                                logger.info(
                                    f"Component {self.component_name} trimmed {trimmed} items"
                                )
                                optimized = True
            else:
                # No manager, use simple thresholds
                if memory_usage > 85:  # Emergency
                    if self.supports_migration:
                        try:
                            migrated = self.component.migrate_to_disk(50.0)
                            if migrated > 0:
                                logger.info(
                                    f"Component {self.component_name} emergency migrated {migrated} items to disk"
                                )
                                optimized = True
                        except Exception as e:
                            logger.warning(f"Emergency migration error: {e}")

                    # If migration didn't work or component doesn't support it, fall back to trim
                    if (
                        not optimized
                        and hasattr(self.component, "trim")
                        and callable(self.component.trim)
                    ):
                        trimmed = self.component.trim(50.0)
                        if trimmed > 0:
                            logger.info(
                                f"Component {self.component_name} emergency trimmed {trimmed} items"
                            )
                            optimized = True

                elif memory_usage > 75:  # Critical
                    if self.supports_migration:
                        try:
                            migrated = self.component.migrate_to_disk(25.0)
                            if migrated > 0:
                                logger.info(
                                    f"Component {self.component_name} critical migrated {migrated} items to disk"
                                )
                                optimized = True
                        except Exception as e:
                            logger.warning(f"Critical migration error: {e}")

                    # Fall back to trim if needed
                    if (
                        not optimized
                        and hasattr(self.component, "trim")
                        and callable(self.component.trim)
                    ):
                        trimmed = self.component.trim(25.0)
                        if trimmed > 0:
                            logger.info(
                                f"Component {self.component_name} critical trimmed {trimmed} items"
                            )
                            optimized = True

        except Exception as e:
            logger.error(f"Error in component memory check: {e}")

        return optimized


def check_memory_usage(
    component_id: str,
    last_check_time: float,
    check_interval: float,
    memory_threshold: Optional[float] = None,
    trim_callback: Optional[Callable[[float], None]] = None,
    migrate_callback: Optional[Callable[[float], int]] = None,
) -> float:
    """
    Unified memory check function that integrates with memory manager.
    Updated to prioritize migration to disk over trimming.

    Args:
        component_id: Unique identifier for the component
        last_check_time: Timestamp of the last memory check
        check_interval: Minimum time between checks in seconds
        memory_threshold: Memory usage threshold to trigger optimization (percentage)
        trim_callback: Optional callback function to trim memory when threshold is exceeded
        migrate_callback: Optional callback function to migrate from memory to disk

    Returns:
        float: Updated last check time
    """
    # If memory manager exists, use its coordination capabilities
    manager = get_memory_manager()
    current_time = time.time()

    # Skip if not enough time has passed - atomic operation
    if (current_time - last_check_time) < check_interval:
        return last_check_time

    # If we have a memory manager, let it decide if checking is necessary
    # This call is designed to be thread-safe without holding locks
    if manager is not None:
        if not manager.should_check_memory(component_id):
            return last_check_time

        # If we have a manager and threshold-based optimization is requested
        if memory_threshold is not None:
            try:
                memory_usage = psutil.virtual_memory().percent

                if memory_usage > memory_threshold:
                    # First try migration if available
                    if migrate_callback is not None:
                        try:
                            migrated = migrate_callback(memory_usage)
                            if migrated > 0:
                                logger.debug(
                                    f"{component_id} migrated {migrated} items to disk"
                                )
                                return current_time
                        except Exception as e:
                            logger.warning(
                                f"Migration callback error for {component_id}: {e}"
                            )

                    # Fall back to trimming if migration is unavailable or didn't free enough memory
                    if trim_callback is not None:
                        try:
                            trim_callback(memory_usage)
                        except Exception as e:
                            logger.warning(
                                f"Trim callback error for {component_id}: {e}"
                            )
            except Exception as e:
                logger.warning(f"Error during managed memory check: {e}")
    else:
        # If no manager but threshold-based optimization is requested
        if memory_threshold is not None:
            try:
                memory_usage = psutil.virtual_memory().percent

                if memory_usage > memory_threshold:
                    # First try migration if available
                    if migrate_callback is not None:
                        try:
                            migrated = migrate_callback(memory_usage)
                            if migrated > 0:
                                return current_time
                        except Exception as e:
                            logger.warning(f"Migration callback error: {e}")

                    # Fall back to trimming
                    if trim_callback is not None:
                        try:
                            trim_callback(memory_usage)
                        except Exception as e:
                            logger.warning(f"Trim callback error: {e}")
            except Exception as e:
                logger.warning(f"Memory check error: {e}")

    # Return updated check time - atomic operation
    return current_time


def optimize_gpu_memory() -> bool:
    """
    Coordinate GPU memory optimization with the memory manager.

    Returns:
        bool: True if optimization was performed
    """
    manager = get_memory_manager()

    # If we have a memory manager, let it handle GPU optimization
    if manager is not None:
        try:
            manager._optimize_gpu_memory()
            return True
        except Exception as e:
            logger.error(f"Error in GPU memory optimization: {e}")

    # Fall back to direct optimization
    try:
        import torch

        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated()
            torch.cuda.empty_cache()
            after = torch.cuda.memory_allocated()

            if before > after:
                logger.info(
                    f"GPU memory optimization freed {(before - after) / 1024**2:.1f} MB"
                )
            return True
    except Exception:
        pass

    return False
