# Keywords4CV Caching Framework
A high-performance caching engine for Keywords4CV and beyond

[![CI](https://github.com/DavidOsipov/k4cv-caching-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/DavidOsipov/k4cv-caching-engine/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/k4cv-caching-engine.svg)](https://badge.fury.io/py/k4cv-caching-engine)
[![Documentation Status](https://readthedocs.org/projects/k4cv-caching-engine/badge/?version=latest)](https://k4cv-caching-engine.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, thread-safe, and extensible caching engine originally developed for Keywords4CV, but designed for general use in Python applications.

## Features

*   **Flexible Backends:** Supports multiple caching backends:
    *   **In-Memory:**  Uses `cachetools` for fast, thread-safe in-memory caching (LRU, TTL).
    *   **Disk:**  Uses `diskcache` for persistent, on-disk caching.
    *   **Hybrid:** Combines in-memory and disk caching for optimal performance.
    *   **Custom:**  Easily extend with your own backends.
*   **Serialization Options:** Supports `pickle` (default) and `msgpack` (optional, for improved performance) for serializing cached objects.
*   **Thread and Process Safety:**  Uses appropriate locking mechanisms to ensure safe concurrent access from multiple threads and processes.
*   **Memory Management:**  Includes a `MemoryManager` to monitor system memory usage and adaptively trim caches to prevent out-of-memory errors.
*   **Extensible Design:**  Provides specialized cache managers for vectors (with optional GPU acceleration using `torch`) and text processing.
*   **Dynamic Configuration:**  Uses `pydantic` for easy configuration and runtime updates.
*   **Security:**  Supports optional encryption of cached data using `cryptography`.
*   **Metrics and Monitoring:**  Built-in metrics tracking (hits, misses, etc.) for integration with monitoring systems (e.g., Prometheus).
*   **Circuit Breaker:**  Includes a circuit breaker to handle failures gracefully when using external resources (e.g., API calls for synonyms).
*   **Comprehensive Testing:**  Includes a thorough test suite with unit tests and performance benchmarks.
*   **Well-Documented:**  Provides clear API documentation and usage examples.

## Installation

**Using pip:**

```bash
pip install k4cv-caching-framework
