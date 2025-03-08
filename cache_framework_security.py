"""
Cache Framework Security Module
==============================

This module provides security features for the Keywords4CV cache framework, including:

1. Secure key derivation using Argon2id (with fallback to PBKDF2)
2. AES-GCM authenticated encryption for cache data
3. MessagePack serialization with optional compression via srsly
4. Secure memory handling and key management
5. Shamir's Secret Sharing for splitting encryption keys
6. Support for asynchronous operations
7. Stream processing for large data sets

Usage Examples
-------------

Basic encryption/decryption:

```python
# Initialize the serializer with configuration
config = {"cache_salt": "your-secure-salt"}
serializer = SecureCacheSerializer(config)

# Encrypt and decrypt binary data
encrypted = serializer.encrypt(b"sensitive data")
decrypted = serializer.decrypt(encrypted)

# Serialize and encrypt Python objects
user_data = {"username": "user123", "preferences": {"theme": "dark"}}
encrypted_data = serializer.secure_serialize(user_data)
original_data = serializer.secure_deserialize(encrypted_data)
```

Key rotation:

```python
# Create a new serializer with rotated key
new_serializer = serializer.rotate_keys(new_secret="new-secret-value", version=2)

# Re-encrypt existing data with new key
new_encrypted = serializer.reencrypt_data(encrypted_data, new_serializer)
```

Key splitting using Shamir's Secret Sharing:

```python
# Split the master key into 5 shares, requiring 3 to reconstruct
key_shares = serializer.split_master_key(shares=5, threshold=3)

# Reconstruct the master key from shares
master_key = SecureCacheSerializer.reconstruct_master_key(key_shares[:3])
```

Stream processing for large data:

```python
# Process large files in chunks
with open('large_file.bin', 'rb') as f:
    # Encrypt in chunks
    encrypted_chunks = list(serializer.encrypt_stream(f))

# Process encrypted chunks
def process_chunk(decrypted_chunk):
    # Do something with each decrypted chunk
    pass

serializer.decrypt_stream(iter(encrypted_chunks), process_chunk)
```

Async operations:

```python
async def process_data():
    encrypted = await serializer.async_encrypt(b"sensitive data")
    decrypted = await serializer.async_decrypt(encrypted)
    return decrypted

# Run in an async context
decrypted_data = await process_data()
```

Security Considerations
----------------------

1. Always use a strong, unique salt for each deployment
2. Store encryption keys securely, consider using environment variables
3. For highest security, use hardware security modules (HSM) for key storage
4. Regularly rotate encryption keys
5. Monitor for suspicious decrypt failures which may indicate tampering

Classes
-------

SecureCacheSerializer: Main class providing encryption, decryption, and serialization services.

Functions
---------

secure_wipe: Securely wipe sensitive data from memory.
derive_key_from_salt: Derive an encryption key from a salt using Argon2id.
get_cache_salt: Get a cache salt from configuration or generate a system-specific one.
get_system_identifier: Generate a unique identifier for the current system.

Constants
---------

DEFAULT_TIME_COST: Default time cost factor for Argon2 (3).
DEFAULT_MEMORY_COST: Default memory usage for Argon2 (64MB).
DEFAULT_PARALLELISM: Default parallelism factor for Argon2 (4).
HASH_LENGTH: Length of hash output in bytes (32).
NONCE_SIZE: Size of nonce in bytes (12).
VERSION: Current version of the encryption format (1).
COMPRESSION_USED_FLAG: Flag indicating compression was used (1).
KEY_VERSION_MASK: Bitmask for extracting key version (0xF0).
ZLIB_AVAILABLE: Boolean indicating if zlib compression is available.

Dependencies
-----------

Required:
- cryptography: For AES-GCM encryption
- srsly: For efficient binary serialization and multiple format support

Optional:
- argon2-cffi: For secure key derivation (highly recommended)
- blake3: For faster hashing
- shamirs-secret-sharing: For key splitting
- zlib: For compression support
- cpuinfo: For detecting hardware acceleration
"""

# Explicitly define the public API
__all__ = [
    # Classes
    "SecureCacheSerializer",
    # Functions
    "secure_wipe",
    "derive_key_from_salt",
    "get_cache_salt",
    "get_system_identifier",
    # Constants
    "DEFAULT_TIME_COST",
    "DEFAULT_MEMORY_COST",
    "DEFAULT_PARALLELISM",
    "HASH_LENGTH",
    "NONCE_SIZE",
    "VERSION",
    "COMPRESSION_USED_FLAG",
    "KEY_VERSION_MASK",
    "ZLIB_AVAILABLE",
]

import os
from typing import Any, Dict, Optional, Tuple, List, BinaryIO, Iterator
import uuid
import platform
import logging
from functools import lru_cache
import concurrent.futures
import asyncio

# Setup logging
logger = logging.getLogger(__name__)

# Handle optional dependencies
try:
    import srsly
except ImportError:
    logger.error("srsly library not installed. Serialization will not work.")
    srsly = None

try:
    import blake3
except ImportError:
    logger.warning("blake3 not installed. Falling back to hashlib.")
    import hashlib

    blake3 = None

try:
    from argon2 import PasswordHasher
    from argon2.low_level import Type
except ImportError:
    logger.error(
        "argon2-cffi library not installed. Secure key derivation will not work."
    )
    PasswordHasher = None

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.exceptions import InvalidTag
except ImportError:
    logger.error("cryptography library not installed. Encryption will not work.")
    AESGCM = None

try:
    import shamirs_secret_sharing as sss
except ImportError:
    logger.warning("shamirs-secret-sharing not installed. Key splitting will not work.")
    sss = None

try:
    import zlib

    ZLIB_AVAILABLE = True
except ImportError:
    logger.warning("zlib not available. Compression will be disabled.")
    ZLIB_AVAILABLE = False

# Constants
DEFAULT_TIME_COST = 3
DEFAULT_MEMORY_COST = 65536  # 64MB memory usage
DEFAULT_PARALLELISM = 4
HASH_LENGTH = 32
NONCE_SIZE = 12
VERSION = 1  # For future key rotation

# Format flags
COMPRESSION_USED_FLAG = 1
KEY_VERSION_MASK = 0xF0  # Upper 4 bits for key version


def get_system_identifier(fallback: str = "default_system") -> str:
    """Generate a unique identifier for this system to avoid using a fixed default salt."""
    try:
        machine_id = str(uuid.getnode())  # MAC address as integer
        system_info = f"{platform.node()}-{platform.system()}"
        return f"{system_info}-{machine_id}"
    except Exception as e:
        logger.warning(f"Failed to get system identifier: {e}. Using fallback.")
        return fallback


def get_cache_salt(config: Dict[str, Any]) -> str:
    """
    Extract the salt from configuration or generate a system-specific one.

    Args:
        config: The configuration dictionary

    Returns:
        str: Salt string for key derivation
    """
    # Check for user-provided salt first
    if "cache_salt" in config:
        return config["cache_salt"]

    # Check for a stored identifier file
    identifier_path = config.get("identifier_path")
    if identifier_path and os.path.exists(identifier_path):
        try:
            with open(identifier_path, "r") as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read identifier file: {e}")

    # Generate and possibly store a new identifier
    identifier = get_system_identifier()
    if identifier_path:
        try:
            os.makedirs(os.path.dirname(identifier_path), exist_ok=True)
            with open(identifier_path, "w") as f:
                f.write(identifier)
        except Exception as e:
            logger.warning(f"Failed to save identifier file: {e}")

    return identifier


@lru_cache(maxsize=4)
def derive_key_from_salt(
    salt: str,
    secret: str = "",
    time_cost: int = DEFAULT_TIME_COST,
    memory_cost: int = DEFAULT_MEMORY_COST,
) -> bytes:
    """
    Derive an encryption key from the salt using Argon2id.

    Args:
        salt: The salt string
        secret: Optional secret for additional security
        time_cost: Time cost factor for Argon2
        memory_cost: Memory cost for Argon2

    Returns:
        bytes: Derived encryption key
    """
    if not salt:
        raise ValueError("Empty salt not allowed for key derivation")

    # Convert string salt to bytes
    salt_bytes = salt.encode("utf-8")

    # Use password if provided, otherwise use a hardcoded value
    password = secret.encode("utf-8") if secret else b"cache_framework_default_secret"

    # Use Argon2id if available, otherwise fallback
    if PasswordHasher:
        try:
            ph = PasswordHasher(
                time_cost=time_cost,
                memory_cost=memory_cost,
                parallelism=DEFAULT_PARALLELISM,
                hash_len=HASH_LENGTH,
                type=Type.ID,
            )

            # Generate a raw hash using low-level API to get the exact bytes we need
            hash_bytes = ph.hash_password_raw(
                password,
                salt_bytes,
                ph.time_cost,
                ph.memory_cost,
                ph.parallelism,
                ph.hash_len,
                ph.type,
            )

            # Generate a unique key fingerprint using BLAKE3 or SHA3
            if blake3:
                key_fingerprint = blake3.blake3(hash_bytes + salt_bytes).digest(32)
            else:
                key_fingerprint = hashlib.sha3_256(hash_bytes + salt_bytes).digest()

            return key_fingerprint
        except Exception as e:
            logger.error(f"Key derivation with Argon2 failed: {e}")
            raise
    else:
        # Fallback to a less secure but available method
        logger.warning("Using fallback key derivation method (less secure)")
        return hashlib.pbkdf2_hmac("sha256", password, salt_bytes, 100000, 32)


def secure_wipe(data: bytearray):
    """Securely wipe sensitive data from memory."""
    try:
        for i in range(len(data)):
            data[i] = 0
    except Exception as e:
        logger.warning(f"Failed to securely wipe data: {e}")


class SecureCacheSerializer:
    """
    Provides encryption and decryption services for cache data using AES-GCM.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.

        Args:
            config: Configuration dictionary containing cache_salt
        """
        self.config = config.copy()  # Store a copy to prevent reference issues

        if AESGCM is None or srsly is None:
            raise ImportError(
                "Required dependencies not available: cryptography and/or srsly"
            )

        # Validate configuration
        self._validate_config()

        self.salt = get_cache_salt(self.config)
        self._key = derive_key_from_salt(
            self.salt,
            secret=self.config.get("cache_secret", ""),
            time_cost=self.config.get("time_cost", DEFAULT_TIME_COST),
            memory_cost=self.config.get("memory_cost", DEFAULT_MEMORY_COST),
        )
        self._init_cipher()
        self.use_compression = self.config.get("use_compression", False)
        self.compression_threshold = self.config.get("compression_threshold", 256)
        self.key_version = self.config.get("key_version", 0) & 0x0F  # Max 15 versions
        self.max_workers = self.config.get(
            "max_workers", None
        )  # For ThreadPoolExecutor

        # Configure serialization format
        self.serialization_format = self.config.get("serialization_format", "msgpack")

    def _init_cipher(self):
        """Initialize cipher with hardware acceleration if available."""
        self._cipher = AESGCM(self._key)

        # Check if hardware acceleration is available
        # The cryptography library automatically uses AES-NI when available,
        # but this check helps with logging/debugging
        hw_accel = False
        try:
            import cpuinfo

            features = cpuinfo.get_cpu_info().get("flags", [])
            hw_accel = "aes" in features
            if hw_accel:
                logger.info("AES-NI hardware acceleration available")
            else:
                logger.info("AES-NI hardware acceleration not available")
        except ImportError:
            logger.debug("cpuinfo not installed, can't detect AES-NI support")
        except Exception as e:
            logger.debug(f"Failed to check for AES-NI: {e}")

    def _validate_config(self):
        """Validate configuration parameters."""
        if "time_cost" in self.config and (
            not isinstance(self.config["time_cost"], int)
            or self.config["time_cost"] < 1
        ):
            raise ValueError("time_cost must be a positive integer")

        if "memory_cost" in self.config and (
            not isinstance(self.config["memory_cost"], int)
            or self.config["memory_cost"] < 8192
        ):
            raise ValueError("memory_cost must be an integer >= 8192")

    def encrypt_batch(self, data_list: List[bytes]) -> List[bytes]:
        """
        Encrypt multiple data items in parallel using a thread pool.

        Args:
            data_list: List of raw data items to encrypt

        Returns:
            List[bytes]: List of encrypted data items
        """
        if not data_list:
            return []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            return list(executor.map(self.encrypt, data_list))

    def decrypt_batch(self, encrypted_data_list: List[bytes]) -> List[bytes]:
        """
        Decrypt multiple data items in parallel using a thread pool.

        Args:
            encrypted_data_list: List of encrypted data items

        Returns:
            List[bytes]: List of decrypted data items
        """
        if not encrypted_data_list:
            return []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            return list(executor.map(self.decrypt, encrypted_data_list))

    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt binary data using AES-GCM with authenticated encryption.

        Args:
            data: Raw data to encrypt

        Returns:
            bytes: Encrypted data with version, flags, and nonce prepended
        """
        if not data:
            raise ValueError("Cannot encrypt empty data")

        # Generate a random nonce for each encryption operation
        nonce = os.urandom(NONCE_SIZE)

        # Include version byte and flags byte for format information
        version_byte = VERSION.to_bytes(1, byteorder="big")
        flags_byte = (
            self.key_version << 4
        ) & KEY_VERSION_MASK  # Store key version in upper 4 bits

        try:
            ciphertext = self._cipher.encrypt(nonce, data, None)
            # Format: [version][flags][nonce][ciphertext]
            return version_byte + bytes([flags_byte]) + nonce + ciphertext
        except Exception as e:
            raise RuntimeError(f"Encryption failed: {str(e)}") from e

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt binary data.

        Args:
            encrypted_data: Data to decrypt with version and nonce

        Returns:
            bytes: Decrypted data

        Raises:
            ValueError: If data is too short or has been tampered with
        """
        # Validate input - need version byte + flags byte + nonce + at least some ciphertext
        if len(encrypted_data) <= (NONCE_SIZE + 2):
            raise ValueError(
                f"Encrypted data too short (must be > {NONCE_SIZE + 2} bytes)"
            )

        # Extract version and validate
        version = int.from_bytes(encrypted_data[0:1], byteorder="big")
        if version != VERSION:
            raise ValueError(f"Unsupported data format version: {version}")

        # Extract flags
        flags = encrypted_data[1]

        # Extract the nonce
        nonce = encrypted_data[2 : NONCE_SIZE + 2]
        ciphertext = encrypted_data[NONCE_SIZE + 2 :]

        # Decrypt and verify authentication
        try:
            return self._cipher.decrypt(nonce, ciphertext, None)
        except InvalidTag:
            raise ValueError("Decryption failed: data may have been tampered with")
        except Exception as e:
            raise RuntimeError(f"Decryption failed: {str(e)}") from e

    def secure_serialize(self, value: Any) -> bytes:
        """
        Serialize and encrypt a Python object using srsly.

        Args:
            value: Python object to serialize and encrypt

        Returns:
            bytes: Encrypted serialized data
        """
        try:
            # Use srsly for serialization
            serialized = srsly.msgpack_dumps(value)
            flags_byte = 0

            # Apply compression if configured and beneficial
            if self.use_compression and len(serialized) > self.compression_threshold:
                try:
                    import zlib

                    compressed = zlib.compress(serialized)
                    # Only use compression if it actually helps
                    if len(compressed) < len(serialized):
                        serialized = compressed
                        flags_byte |= COMPRESSION_USED_FLAG
                except ImportError:
                    logger.warning("Compression requested but zlib not available")
                except Exception as e:
                    logger.warning(f"Compression failed: {e}")

            # Include version byte and flags byte for format information
            version_byte = VERSION.to_bytes(1, byteorder="big")

            # Generate a random nonce for each encryption operation
            nonce = os.urandom(NONCE_SIZE)

            # Encrypt the data
            ciphertext = self._cipher.encrypt(nonce, serialized, None)

            # Format: [version][flags][nonce][ciphertext]
            return version_byte + bytes([flags_byte]) + nonce + ciphertext

        except Exception as e:
            raise RuntimeError(f"Serialization failed: {str(e)}") from e

    def secure_deserialize(
        self, encrypted_data: bytes, schema_validator: callable = None
    ) -> Any:
        """
        Decrypt and deserialize to a Python object with optional schema validation.

        Args:
            encrypted_data: Encrypted serialized data
            schema_validator: Optional callable that validates the deserialized structure
                              Should raise ValueError if validation fails

        Returns:
            Any: Deserialized Python object

        Raises:
            ValueError: If data is corrupt, tampered with, or fails schema validation
        """
        if len(encrypted_data) <= (NONCE_SIZE + 2):
            raise ValueError(
                f"Encrypted data too short (must be > {NONCE_SIZE + 2} bytes)"
            )

        try:
            # Extract version and validate
            version = int.from_bytes(encrypted_data[0:1], byteorder="big")
            if version != VERSION:
                raise ValueError(f"Unsupported data format version: {version}")

            # Extract flags
            flags = encrypted_data[1]
            compression_used = bool(flags & COMPRESSION_USED_FLAG)

            # Extract the nonce
            nonce = encrypted_data[2 : NONCE_SIZE + 2]
            ciphertext = encrypted_data[NONCE_SIZE + 2 :]

            # Decrypt the data - AES-GCM already verifies integrity
            decrypted = self._cipher.decrypt(nonce, ciphertext, None)

            # Apply decompression if flag indicates it was used
            if compression_used:
                try:
                    import zlib

                    decrypted = zlib.decompress(decrypted)
                except ImportError:
                    raise RuntimeError("Data was compressed but zlib is not available")
                except zlib.error as e:
                    raise ValueError(f"Failed to decompress data: {str(e)}")

            # Deserialize using srsly
            deserialized_data = srsly.msgpack_loads(decrypted)

            # Apply schema validation if provided
            if schema_validator is not None:
                try:
                    schema_validator(deserialized_data)
                except Exception as e:
                    raise ValueError(f"Schema validation failed: {str(e)}")

            return deserialized_data

        except InvalidTag:
            # This is raised by AES-GCM when authentication fails
            raise ValueError("Decryption failed: data integrity check failed")
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Deserialization failed: {str(e)}") from e

    def rotate_keys(
        self, new_secret: str, version: int = None
    ) -> "SecureCacheSerializer":
        """
        Create a new serializer with a rotated key for re-encrypting cache data.

        Args:
            new_secret: New secret for key derivation
            version: New key version (1-15, default increments current version)

        Returns:
            SecureCacheSerializer: New serializer with the rotated key
        """
        if version is not None and (
            not isinstance(version, int) or version < 0 or version > 15
        ):
            raise ValueError("Key version must be an integer between 0 and 15")

        # Create new config with incremented version if not specified
        new_config = self.config.copy()
        new_config["cache_secret"] = new_secret
        new_config["key_version"] = (
            version if version is not None else (self.key_version + 1) % 16
        )

        return SecureCacheSerializer(new_config)

    def reencrypt_data(
        self, encrypted_data: bytes, new_serializer: "SecureCacheSerializer"
    ) -> bytes:
        """
        Re-encrypt data using a new key.

        Args:
            encrypted_data: Data encrypted with this serializer's key
            new_serializer: Serializer with the new key

        Returns:
            bytes: Data encrypted with the new key
        """
        # Decrypt with current key
        decrypted = self.decrypt(encrypted_data)

        # Encrypt with new key
        return new_serializer.encrypt(decrypted)

    def split_master_key(self, shares: int, threshold: int) -> List[bytes]:
        """
        Split the master key into multiple shares using Shamir's Secret Sharing.

        Args:
            shares: Number of key shares to create
            threshold: Minimum number of shares required to reconstruct the key

        Returns:
            List[bytes]: The key shares
        """
        if sss is None:
            raise ImportError(
                "shamirs-secret-sharing package required for key splitting"
            )

        if threshold > shares:
            raise ValueError("Threshold cannot exceed the number of shares")

        if threshold < 2:
            raise ValueError("Threshold must be at least 2")

        try:
            # Create key shares
            key_shares = sss.share(self._key, shares=shares, threshold=threshold)
            return key_shares
        except Exception as e:
            raise RuntimeError(f"Key splitting failed: {str(e)}") from e

    @staticmethod
    def reconstruct_master_key(key_shares: List[bytes]) -> bytes:
        """
        Reconstruct the master key from shares.

        Args:
            key_shares: List of key shares (need at least threshold shares)

        Returns:
            bytes: The reconstructed master key
        """
        if sss is None:
            raise ImportError(
                "shamirs-secret-sharing package required for key reconstruction"
            )

        try:
            return sss.combine(key_shares)
        except Exception as e:
            raise ValueError(f"Key reconstruction failed: {str(e)}") from e

    async def async_encrypt(self, data: bytes) -> bytes:
        """
        Async version of encrypt method.

        Args:
            data: Raw data to encrypt

        Returns:
            bytes: Encrypted data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encrypt, data)

    async def async_decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Async version of decrypt method.

        Args:
            encrypted_data: Data to decrypt

        Returns:
            bytes: Decrypted data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.decrypt, encrypted_data)

    async def async_secure_serialize(self, value: Any) -> bytes:
        """
        Async version of secure_serialize method.

        Args:
            value: Python object to serialize and encrypt

        Returns:
            bytes: Encrypted serialized data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.secure_serialize, value)

    async def async_secure_deserialize(self, encrypted_data: bytes) -> Any:
        """
        Async version of secure_deserialize method.

        Args:
            encrypted_data: Encrypted serialized data

        Returns:
            Any: Deserialized Python object
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.secure_deserialize, encrypted_data)

    def encrypt_stream(
        self, data_stream: BinaryIO, chunk_size: int = 8192
    ) -> Iterator[bytes]:
        """
        Process a data stream in chunks to minimize memory usage.

        Args:
            data_stream: File-like object containing data to encrypt
            chunk_size: Size of chunks to process

        Returns:
            Iterator[bytes]: Iterator of encrypted chunks
        """
        # First chunk includes header with version, flags
        first_chunk = True

        # Generate a random nonce for the entire stream (same across chunks)
        nonce = os.urandom(NONCE_SIZE)

        # Add version and flags info to first chunk only
        version_byte = VERSION.to_bytes(1, byteorder="big")
        flags_byte = (self.key_version << 4) & KEY_VERSION_MASK

        # Associated data to ensure stream integrity
        stream_id = os.urandom(16)  # Unique ID for this stream
        position = 0  # Track position in stream

        while True:
            chunk = data_stream.read(chunk_size)
            if not chunk:
                break

            # Create associated data that includes position to prevent chunk reordering
            associated_data = stream_id + position.to_bytes(8, byteorder="big")
            position += 1

            try:
                encrypted_chunk = self._cipher.encrypt(nonce, chunk, associated_data)

                if first_chunk:
                    # First chunk includes header and stream ID
                    yield (
                        version_byte
                        + bytes([flags_byte])
                        + nonce
                        + stream_id
                        + encrypted_chunk
                    )
                    first_chunk = False
                else:
                    # Subsequent chunks only have the encrypted data
                    yield encrypted_chunk
            except Exception as e:
                raise RuntimeError(
                    f"Stream encryption failed at position {position - 1}: {str(e)}"
                ) from e

    def decrypt_stream(
        self, encrypted_chunks: Iterator[bytes], chunk_handler: callable
    ) -> None:
        """
        Decrypt a stream of encrypted chunks and process with handler function.

        Args:
            encrypted_chunks: Iterator providing encrypted chunks
            chunk_handler: Function that processes each decrypted chunk
        """
        # Get first chunk to extract header information
        try:
            first_chunk = next(encrypted_chunks)
        except StopIteration:
            raise ValueError("Empty encrypted stream")

        if len(first_chunk) <= (
            NONCE_SIZE + 2 + 16
        ):  # Version + flags + nonce + stream_id
            raise ValueError("First chunk too small to contain header information")

        # Extract header info
        version = int.from_bytes(first_chunk[0:1], byteorder="big")
        if version != VERSION:
            raise ValueError(f"Unsupported data format version: {version}")

        flags = first_chunk[1]
        key_version = (flags & KEY_VERSION_MASK) >> 4

        # Check if this key can decrypt data from the stored key version
        if key_version != self.key_version and not self.config.get(
            "allow_version_mismatch", False
        ):
            raise ValueError(
                f"Key version mismatch: data is v{key_version}, current key is v{self.key_version}"
            )

        # Extract nonce and stream ID
        nonce = first_chunk[2 : NONCE_SIZE + 2]
        stream_id = first_chunk[NONCE_SIZE + 2 : NONCE_SIZE + 18]
        first_data = first_chunk[NONCE_SIZE + 18 :]

        # Process first chunk's data
        position = 0
        associated_data = stream_id + position.to_bytes(8, byteorder="big")

        try:
            decrypted = self._cipher.decrypt(nonce, first_data, associated_data)
            chunk_handler(decrypted)
            position += 1
        except InvalidTag:
            raise ValueError("Decryption failed: data may have been tampered with")

        # Process remaining chunks
        for chunk in encrypted_chunks:
            if not chunk:
                continue

            associated_data = stream_id + position.to_bytes(8, byteorder="big")
            try:
                decrypted = self._cipher.decrypt(nonce, chunk, associated_data)
                chunk_handler(decrypted)
                position += 1
            except InvalidTag:
                raise ValueError(
                    f"Decryption failed at chunk {position}: data may have been tampered with"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Stream decryption failed at chunk {position}: {str(e)}"
                ) from e

    def __del__(self):
        """Clean up sensitive data when the object is destroyed."""
        if hasattr(self, "_key"):
            # Convert to bytearray to allow wiping
            key_bytes = bytearray(self._key)
            secure_wipe(key_bytes)
