"""
Memory manager for Keywords4CV cache framework.
Coordinates memory usage across different cache components to ensure
efficient operation on memory-constrained systems (e.g., 8GB RAM).
"""

import logging
import time
import gc
import psutil
import threading
import weakref
from typing import Dict, Any, List, Set, Optional, Callable, Tuple

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Centralized memory manager for coordinating memory usage across
    different cache components.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the memory manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.hardware_limits = config.get("hardware_limits", {})

        # Set memory thresholds
        self.warning_threshold = self.hardware_limits.get("memory_warning_percent", 65)
        self.critical_threshold = self.hardware_limits.get("max_ram_usage_percent", 75)
        self.emergency_threshold = self.hardware_limits.get("emergency_percent", 85)

        # Track registered cache components
        self.cache_components = weakref.WeakSet()

        # Track components that support migration to disk
        self._migration_capable_components = weakref.WeakSet()

        # Component registry with priorities
        self._components_by_priority = {
            "high": weakref.WeakSet(),  # Critical components like vector caches
            "medium": weakref.WeakSet(),  # Standard caches
            "low": weakref.WeakSet(),  # Less critical components
        }

        # Track migration-capable components by priority
        self._migration_components_by_priority = {
            "high": weakref.WeakSet(),
            "medium": weakref.WeakSet(),
            "low": weakref.WeakSet(),
        }

        # Set up monitoring thread
        self.monitoring_interval = config.get("caching", {}).get(
            "memory_check_interval", 60
        )
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()

        # Optimization coordination
        self._last_optimization = 0
        self._optimization_cooling_period = 10  # Seconds between optimizations

        # Set up lock with proper hierarchy
        try:
            from lock_utils import create_component_lock

            self._optimization_lock = create_component_lock(
                "memory_manager_optimization"
            )
        except ImportError:
            self._optimization_lock = threading.RLock()

        self._optimization_in_progress = False

        # Add coordination tracking to prevent cascading optimizations
        self._recent_component_checks = {}
        self._component_check_cooldown = 5  # Seconds between checks for a component

        # Protect the check registry with a separate lock to avoid deadlocks
        try:
            from lock_utils import create_component_lock

            self._component_check_lock = create_component_lock("memory_manager_checks")
        except ImportError:
            self._component_check_lock = threading.RLock()

        # Initialize memory monitoring
        self.start_monitoring()

        # Stats tracking for optimization methods
        self._optimization_stats = {
            "migrations": 0,
            "trims": 0,
            "migrated_items": 0,
            "trimmed_items": 0,
        }

        logger.info(
            f"Memory manager initialized (Warning: {self.warning_threshold}%, "
            f"Critical: {self.critical_threshold}%, "
            f"Emergency: {self.emergency_threshold}%)"
        )

    def register_component(self, component: Any, priority: str = "medium") -> None:
        """
        Register a cache component to be managed.

        Args:
            component: Cache component with trim() method
            priority: Priority level ("high", "medium", "low") for optimization order
        """
        if hasattr(component, "trim") and callable(getattr(component, "trim")):
            self.cache_components.add(component)

            # Add to priority-based registry
            if priority in self._components_by_priority:
                self._components_by_priority[priority].add(component)
            else:
                self._components_by_priority["medium"].add(component)

            # Check if component supports migration to disk
            if hasattr(component, "migrate_to_disk") and callable(
                getattr(component, "migrate_to_disk")
            ):
                self._migration_capable_components.add(component)

                # Add to migration-capable priority registry
                if priority in self._migration_components_by_priority:
                    self._migration_components_by_priority[priority].add(component)
                else:
                    self._migration_components_by_priority["medium"].add(component)

                logger.debug(
                    f"Registered migration-capable component {component.__class__.__name__} "
                    f"with memory manager (priority: {priority})"
                )
            else:
                logger.debug(
                    f"Registered trim-only component {component.__class__.__name__} "
                    f"with memory manager (priority: {priority})"
                )
        else:
            logger.warning(
                f"Component {component.__class__.__name__} missing trim() method"
            )

    def start_monitoring(self) -> None:
        """Start the memory monitoring thread."""
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            return

        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_memory, daemon=True, name="MemoryMonitor"
        )
        self.monitor_thread.start()
        logger.debug("Memory monitoring started")

    def stop(self) -> None:
        """Stop the memory monitoring thread."""
        if self.monitor_thread is not None:
            self.stop_monitoring.set()
            self.monitor_thread.join(timeout=2.0)
            logger.debug("Memory monitoring stopped")

    def _monitor_memory(self) -> None:
        """Monitor memory usage and trigger optimizations when needed."""
        while not self.stop_monitoring.is_set():
            try:
                # Check current memory usage
                memory_usage = psutil.virtual_memory().percent

                # Take action based on memory usage level
                if memory_usage >= self.emergency_threshold:
                    logger.warning(f"EMERGENCY memory usage: {memory_usage}%")
                    self.optimize_memory(level="emergency")
                elif memory_usage >= self.critical_threshold:
                    logger.info(f"Critical memory usage: {memory_usage}%")
                    self.optimize_memory(level="critical")
                elif memory_usage >= self.warning_threshold:
                    logger.debug(f"High memory usage: {memory_usage}%")
                    self.optimize_memory(level="standard")

            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")

            # Sleep until next check
            self.stop_monitoring.wait(self.monitoring_interval)

    def optimize_memory(self, level: str = "standard", force: bool = False) -> bool:
        """
        Perform memory optimization at the specified level.
        Returns True if optimization was performed, False if skipped.

        Args:
            level: Optimization level ("standard", "critical", "emergency")
            force: Force optimization even during cooling period

        Returns:
            bool: True if optimization performed, False if skipped
        """
        # Check cooling period unless forced
        current_time = time.time()
        if (
            not force
            and (current_time - self._last_optimization)
            < self._optimization_cooling_period
        ):
            logger.debug("Optimization skipped: cooling period active")
            return False

        # Use lock to prevent concurrent optimizations
        if not self._optimization_lock.acquire(blocking=False):
            logger.debug("Optimization skipped: another optimization in progress")
            return False

        try:
            self._optimization_in_progress = True

            # Determine trim percentages based on level
            if level == "emergency":
                self._emergency_optimization()
            elif level == "critical":
                self._critical_optimization()
            else:
                self._standard_optimization()

            # Update optimization timestamp
            self._last_optimization = time.time()
            return True

        finally:
            self._optimization_in_progress = False
            self._optimization_lock.release()

    def _standard_optimization(self) -> None:
        """Perform standard memory optimization (moderate trimming)."""
        # First attempt migration for components that support it
        migrated = self._migrate_by_priority(
            {
                "high": 5.0,  # Migrate 5% from high priority components
                "medium": 10.0,  # Migrate 10% from medium priority components
                "low": 20.0,  # Migrate 20% from low priority components
            }
        )

        # Then trim components that don't support migration or if migration wasn't sufficient
        if migrated == 0 or psutil.virtual_memory().percent > self.warning_threshold:
            self._trim_by_priority(
                {
                    "high": 5.0,  # Trim 5% from high priority components
                    "medium": 10.0,  # Trim 10% from medium priority components
                    "low": 15.0,  # Trim 15% from low priority components
                }
            )

        gc.collect(0)  # Collect only the youngest generation

    def _critical_optimization(self) -> None:
        """Perform critical memory optimization (aggressive trimming)."""
        # First try migration with higher percentages
        migrated = self._migrate_by_priority(
            {
                "high": 15.0,  # Migrate 15% from high priority components
                "medium": 25.0,  # Migrate 25% from medium priority components
                "low": 40.0,  # Migrate 40% from low priority components
            }
        )

        # Then trim as needed
        if migrated == 0 or psutil.virtual_memory().percent > self.critical_threshold:
            self._trim_by_priority(
                {
                    "high": 15.0,  # Trim 15% from high priority components
                    "medium": 25.0,  # Trim 25% from medium priority components
                    "low": 40.0,  # Trim 40% from low priority components
                }
            )

        gc.collect(1)  # Collect the first two generations

        # Try to release GPU memory if available
        self._optimize_gpu_memory()

    def _emergency_optimization(self) -> None:
        """
        Perform emergency memory optimization when system is running out of memory.
        Very aggressive trimming and cleanup.
        """
        # First try aggressive migration
        migrated = self._migrate_by_priority(
            {
                "high": 30.0,  # Migrate 30% from high priority components
                "medium": 50.0,  # Migrate 50% from medium priority components
                "low": 75.0,  # Migrate 75% from low priority components
            }
        )

        # Then perform aggressive trimming regardless
        self._trim_by_priority(
            {
                "high": 30.0,  # Trim 30% from high priority components
                "medium": 50.0,  # Trim 50% from medium priority components
                "low": 75.0,  # Trim 75% from low priority components
            }
        )

        # Force full garbage collection
        gc.collect(2)

        # Try to release GPU memory if available
        self._optimize_gpu_memory()

        logger.warning("Emergency memory optimization completed")

    def _migrate_by_priority(self, migrate_percents: Dict[str, float]) -> int:
        """
        Migrate data from memory to disk for components that support it.

        Args:
            migrate_percents: Dictionary mapping priority levels to migration percentages

        Returns:
            int: Total number of entries migrated
        """
        total_migrated = 0

        # Process each priority level
        for priority, percent in migrate_percents.items():
            components = list(self._migration_components_by_priority.get(priority, []))

            for component in components:
                try:
                    if hasattr(component, "migrate_to_disk") and callable(
                        component.migrate_to_disk
                    ):
                        migrated = component.migrate_to_disk(percent)
                        total_migrated += migrated
                        self._optimization_stats["migrated_items"] += migrated
                except Exception as e:
                    logger.error(
                        f"Error migrating component {component.__class__.__name__} to disk: {e}"
                    )

        if total_migrated > 0:
            self._optimization_stats["migrations"] += 1
            logger.info(
                f"Migrated {total_migrated} entries to disk across priority levels"
            )

        return total_migrated

    def _trim_by_priority(self, trim_percents: Dict[str, float]) -> int:
        """
        Trim registered cache components based on priority levels.

        Args:
            trim_percents: Dictionary mapping priority levels to trim percentages

        Returns:
            int: Total number of entries trimmed
        """
        total_trimmed = 0

        # Process each priority level
        for priority, percent in trim_percents.items():
            components = list(self._components_by_priority.get(priority, []))

            for component in components:
                try:
                    if hasattr(component, "trim") and callable(component.trim):
                        # Skip components we've already migrated to disk
                        if (
                            component in self._migration_capable_components
                            and hasattr(component, "migrate_to_disk")
                            and callable(component.migrate_to_disk)
                        ):
                            continue

                        trimmed = component.trim(percent)
                        total_trimmed += trimmed
                        self._optimization_stats["trimmed_items"] += trimmed
                except Exception as e:
                    logger.error(
                        f"Error trimming component {component.__class__.__name__}: {e}"
                    )

        if total_trimmed > 0:
            self._optimization_stats["trims"] += 1
            logger.info(f"Trimmed {total_trimmed} entries across priority levels")

        return total_trimmed

    def _trim_components(self, trim_percent: float) -> int:
        """
        Trim registered cache components.

        Args:
            trim_percent: Percentage of entries to trim

        Returns:
            int: Total number of entries trimmed
        """
        total_trimmed = 0

        # Make a copy of the weakrefs to avoid modification during iteration
        components = list(self.cache_components)

        for component in components:
            try:
                if hasattr(component, "trim") and callable(component.trim):
                    trimmed = component.trim(trim_percent)
                    total_trimmed += trimmed
            except Exception as e:
                logger.error(
                    f"Error trimming component {component.__class__.__name__}: {e}"
                )

        if total_trimmed > 0:
            logger.info(
                f"Trimmed {total_trimmed} entries from {len(components)} components"
            )

        return total_trimmed

    def _optimize_gpu_memory(self) -> None:
        """Try to optimize GPU memory usage if available."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
        except ImportError:
            pass  # GPU optimization not available

    def should_check_memory(self, component_id: str) -> bool:
        """
        Determine if a component should perform its own memory check.
        Updated to avoid lock conflicts.
        """
        current_time = time.time()

        # If optimization in progress, block all checks - atomic read
        if self._optimization_in_progress:
            return False

        # Check cooling period after optimization - atomic read
        if (current_time - self._last_optimization) < self._optimization_cooling_period:
            return False

        # For component-specific cooldown, use a separate lock to avoid deadlocks
        with self._component_check_lock:
            # Check component-specific cooldown
            last_check = self._recent_component_checks.get(component_id, 0)
            if (current_time - last_check) < self._component_check_cooldown:
                return False

            # Update last check time for this component
            self._recent_component_checks[component_id] = current_time

        # Allow checks but with reduced frequency based on component ID hash
        # This spreads memory checks across time instead of having all components
        # check at the same time
        component_hash = hash(component_id) % 100
        seconds_since_last_check = current_time % 60

        # Distribute checks throughout the minute based on hash
        return component_hash <= (seconds_since_last_check * 1.7)

    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get current memory status.

        Returns:
            Dict with memory usage statistics
        """
        try:
            vm = psutil.virtual_memory()
            status = {
                "total_mb": vm.total / (1024 * 1024),
                "available_mb": vm.available / (1024 * 1024),
                "used_mb": vm.used / (1024 * 1024),
                "percent": vm.percent,
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold,
                "emergency_threshold": self.emergency_threshold,
                "last_optimization": self._last_optimization,
                "cooling_period": self._optimization_cooling_period,
                "optimization_in_progress": self._optimization_in_progress,
                "optimization_stats": self._optimization_stats.copy(),
            }

            # Add migration capability info
            status["migration_capable_components"] = len(
                self._migration_capable_components
            )
            status["trim_only_components"] = len(self.cache_components) - len(
                self._migration_capable_components
            )

            return status
        except Exception as e:
            logger.error(f"Error getting memory status: {e}")
            return {"error": str(e)}

    def component_memory_check(
        self, component_id: str, current_usage: float
    ) -> Tuple[bool, float]:
        """
        Centralized memory check for components.
        Updated to avoid lock conflicts.
        """
        # If we shouldn't check or optimization is in progress/cooling, skip
        # This call is designed to be thread-safe
        if not self.should_check_memory(component_id):
            return False, 0.0

        # If memory usage is already known and high, suggest optimization
        if current_usage >= self.emergency_threshold:
            return True, 50.0
        elif current_usage >= self.critical_threshold:
            return True, 25.0
        elif current_usage >= self.warning_threshold:
            return True, 10.0

        # If memory usage wasn't passed or isn't high, check system memory
        try:
            memory_usage = psutil.virtual_memory().percent

            if memory_usage >= self.emergency_threshold:
                return True, 50.0
            elif memory_usage >= self.critical_threshold:
                return True, 25.0
            elif memory_usage >= self.warning_threshold:
                return True, 10.0
        except Exception:
            pass

        # Default: no optimization needed
        return False, 0.0


# Example usage
def create_memory_manager(config: Dict[str, Any]) -> MemoryManager:
    """
    Create and return a memory manager instance.

    Args:
        config: Configuration dictionary

    Returns:
        Configured MemoryManager instance
    """
    return MemoryManager(config)
