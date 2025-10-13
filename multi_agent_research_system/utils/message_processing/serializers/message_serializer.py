"""
Message Serializer and Persistence - Comprehensive Message Serialization

This module provides sophisticated message serialization and persistence capabilities
with multiple format support, compression, and efficient storage mechanisms.

Key Features:
- Multiple serialization formats (JSON, Pickle, MessagePack, YAML)
- Compression support for efficient storage
- Batch serialization and deserialization
- Message persistence with file and database backends
- Versioning and compatibility handling
- Integrity validation and checksum verification
- Encrypted serialization support for sensitive messages
- Streaming serialization for large messages
"""

import asyncio
import json
import pickle
import hashlib
import gzip
import lzma
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO, TextIO
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import sqlite3
import threading

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from ..core.message_types import RichMessage, EnhancedMessageType, MessagePriority


class SerializationFormat(Enum):
    """Supported serialization formats."""

    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    YAML = "yaml"


class CompressionType(Enum):
    """Supported compression types."""

    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"


class PersistenceBackend(Enum):
    """Persistence backend types."""

    FILE = "file"
    SQLITE = "sqlite"
    CUSTOM = "custom"


@dataclass
class SerializationConfig:
    """Configuration for message serialization."""

    default_format: SerializationFormat = SerializationFormat.JSON
    default_compression: CompressionType = CompressionType.NONE
    enable_compression: bool = True
    compression_threshold: int = 1024
    enable_integrity_check: bool = True
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    enable_versioning: bool = True
    current_version: str = "1.0"
    max_serialized_size: int = 100 * 1024 * 1024  # 100MB
    pretty_print: bool = True
    include_metadata: bool = True


@dataclass
class SerializedMessage:
    """Serialized message with metadata."""

    data: bytes
    format: SerializationFormat
    compression: CompressionType
    original_size: int
    compressed_size: int
    checksum: str
    version: str
    timestamp: datetime
    message_id: str
    message_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageSerializer:
    """Advanced message serializer with multiple format support."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize message serializer with configuration."""
        self.config = self._create_serialization_config(config or {})
        self.logger = logging.getLogger(__name__)

        # Initialize encryption if enabled
        self.cipher = None
        if self.config.enable_encryption and CRYPTO_AVAILABLE:
            self._initialize_encryption()

        # Serialization statistics
        self.serialization_stats = {
            "total_serialized": 0,
            "total_deserialized": 0,
            "serialization_time": 0.0,
            "deserialization_time": 0.0,
            "compression_saves": 0,
            "total_bytes_saved": 0,
            "by_format": {},
            "errors": 0
        }

        # Thread safety
        self.lock = threading.RLock()

    def _create_serialization_config(self, config: Dict[str, Any]) -> SerializationConfig:
        """Create serialization configuration from settings."""
        return SerializationConfig(
            default_format=SerializationFormat(config.get("default_format", "json")),
            default_compression=CompressionType(config.get("default_compression", "none")),
            enable_compression=config.get("enable_compression", True),
            compression_threshold=config.get("compression_threshold", 1024),
            enable_integrity_check=config.get("enable_integrity_check", True),
            enable_encryption=config.get("enable_encryption", False),
            encryption_key=config.get("encryption_key"),
            enable_versioning=config.get("enable_versioning", True),
            current_version=config.get("current_version", "1.0"),
            max_serialized_size=config.get("max_serialized_size", 100 * 1024 * 1024),
            pretty_print=config.get("pretty_print", True),
            include_metadata=config.get("include_metadata", True)
        )

    def _initialize_encryption(self):
        """Initialize encryption cipher."""
        if not self.config.encryption_key:
            raise ValueError("Encryption key is required when encryption is enabled")

        try:
            # Ensure key is proper length for Fernet (32 bytes base64 encoded)
            key = self.config.encryption_key.encode()
            if len(key) != 44:  # Fernet key is 44 bytes base64 encoded
                # Derive proper key
                import hashlib
                import base64
                key = base64.urlsafe_b64encode(hashlib.sha256(key).digest())

            self.cipher = Fernet(key)
            self.logger.info("Encryption initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {str(e)}")
            raise

    async def serialize(self, message: RichMessage,
                       format: Optional[SerializationFormat] = None,
                       compression: Optional[CompressionType] = None) -> SerializedMessage:
        """Serialize a message to bytes."""
        start_time = datetime.now()

        try:
            with self.lock:
                # Use defaults if not specified
                if format is None:
                    format = self.config.default_format
                if compression is None:
                    compression = self.config.default_compression if self.config.enable_compression else CompressionType.NONE

                # Convert message to dictionary
                message_dict = message.to_dict()

                # Add serialization metadata
                if self.config.include_metadata:
                    message_dict["_serialization_metadata"] = {
                        "serialized_at": datetime.now().isoformat(),
                        "serializer_version": self.config.current_version,
                        "format": format.value,
                        "compression": compression.value
                    }

                # Serialize to bytes
                serialized_data = await self._serialize_to_bytes(message_dict, format)

                original_size = len(serialized_data)

                # Apply compression if enabled
                if compression != CompressionType.NONE and original_size > self.config.compression_threshold:
                    compressed_data = await self._compress_data(serialized_data, compression)
                    if len(compressed_data) < original_size:
                        serialized_data = compressed_data
                        self.serialization_stats["compression_saves"] += 1
                        self.serialization_stats["total_bytes_saved"] += (original_size - len(serialized_data))

                # Apply encryption if enabled
                if self.config.enable_encryption and self.cipher:
                    serialized_data = self.cipher.encrypt(serialized_data)

                # Calculate checksum
                checksum = ""
                if self.config.enable_integrity_check:
                    checksum = hashlib.sha256(serialized_data).hexdigest()

                # Check size limits
                if len(serialized_data) > self.config.max_serialized_size:
                    raise ValueError(f"Serialized message size ({len(serialized_data)}) exceeds limit ({self.config.max_serialized_size})")

                # Create serialized message object
                serialized_message = SerializedMessage(
                    data=serialized_data,
                    format=format,
                    compression=compression,
                    original_size=original_size,
                    compressed_size=len(serialized_data),
                    checksum=checksum,
                    version=self.config.current_version,
                    timestamp=datetime.now(),
                    message_id=message.id,
                    message_type=message.message_type.value,
                    metadata={
                        "processing_time": message.performance_metrics.get("total_processing_time", 0.0),
                        "quality_score": message.metadata.quality_score,
                        "session_id": message.session_id,
                        "agent_name": message.agent_name
                    }
                )

                # Update statistics
                processing_time = (datetime.now() - start_time).total_seconds()
                self._update_serialization_stats(format, processing_time, True)

                return serialized_message

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_serialization_stats(format, processing_time, False)
            self.logger.error(f"Serialization failed for message {message.id}: {str(e)}")
            raise

    async def deserialize(self, serialized_message: SerializedMessage) -> RichMessage:
        """Deserialize bytes to a message."""
        start_time = datetime.now()

        try:
            with self.lock:
                data = serialized_message.data

                # Verify checksum
                if self.config.enable_integrity_check and serialized_message.checksum:
                    calculated_checksum = hashlib.sha256(data).hexdigest()
                    if calculated_checksum != serialized_message.checksum:
                        raise ValueError("Checksum verification failed - data may be corrupted")

                # Apply decryption if enabled
                if self.config.enable_encryption and self.cipher:
                    data = self.cipher.decrypt(data)

                # Apply decompression
                if serialized_message.compression != CompressionType.NONE:
                    data = await self._decompress_data(data, serialized_message.compression)

                # Deserialize from bytes
                message_dict = await self._deserialize_from_bytes(data, serialized_message.format)

                # Remove serialization metadata
                if self.config.include_metadata and "_serialization_metadata" in message_dict:
                    del message_dict["_serialization_metadata"]

                # Create RichMessage object
                message = RichMessage.from_dict(message_dict)

                # Update statistics
                processing_time = (datetime.now() - start_time).total_seconds()
                self._update_deserialization_stats(serialized_message.format, processing_time, True)

                return message

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_deserialization_stats(serialized_message.format, processing_time, False)
            self.logger.error(f"Deserialization failed: {str(e)}")
            raise

    async def serialize_to_bytes(self, message: RichMessage,
                                format: Optional[SerializationFormat] = None) -> bytes:
        """Serialize message directly to bytes."""
        serialized = await self.serialize(message, format)
        return serialized.data

    async def deserialize_from_bytes(self, data: bytes,
                                   format: SerializationFormat,
                                   compression: CompressionType = CompressionType.NONE,
                                   checksum: str = "") -> RichMessage:
        """Deserialize bytes directly to message."""
        serialized_message = SerializedMessage(
            data=data,
            format=format,
            compression=compression,
            original_size=len(data),
            compressed_size=len(data),
            checksum=checksum,
            version=self.config.current_version,
            timestamp=datetime.now(),
            message_id="",
            message_type=""
        )

        return await self.deserialize(serialized_message)

    async def _serialize_to_bytes(self, data: Dict[str, Any], format: SerializationFormat) -> bytes:
        """Serialize dictionary to bytes using specified format."""
        if format == SerializationFormat.JSON:
            if self.config.pretty_print:
                return json.dumps(data, indent=2, default=str, ensure_ascii=False).encode('utf-8')
            else:
                return json.dumps(data, default=str, ensure_ascii=False).encode('utf-8')

        elif format == SerializationFormat.PICKLE:
            return pickle.dumps(data)

        elif format == SerializationFormat.MSGPACK:
            if not MSGPACK_AVAILABLE:
                raise ImportError("MessagePack is not available. Install with: pip install msgpack")
            return msgpack.packb(data, default=str)

        elif format == SerializationFormat.YAML:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is not available. Install with: pip install pyyaml")
            return yaml.dump(data, default_flow_style=False, allow_unicode=True).encode('utf-8')

        else:
            raise ValueError(f"Unsupported serialization format: {format}")

    async def _deserialize_from_bytes(self, data: bytes, format: SerializationFormat) -> Dict[str, Any]:
        """Deserialize bytes to dictionary using specified format."""
        if format == SerializationFormat.JSON:
            return json.loads(data.decode('utf-8'))

        elif format == SerializationFormat.PICKLE:
            return pickle.loads(data)

        elif format == SerializationFormat.MSGPACK:
            if not MSGPACK_AVAILABLE:
                raise ImportError("MessagePack is not available")
            return msgpack.unpackb(data, raw=False, strict_map_key=False)

        elif format == SerializationFormat.YAML:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is not available")
            return yaml.safe_load(data.decode('utf-8'))

        else:
            raise ValueError(f"Unsupported serialization format: {format}")

    async def _compress_data(self, data: bytes, compression: CompressionType) -> bytes:
        """Compress data using specified compression type."""
        if compression == CompressionType.GZIP:
            return gzip.compress(data)

        elif compression == CompressionType.LZMA:
            return lzma.compress(data)

        else:
            return data

    async def _decompress_data(self, data: bytes, compression: CompressionType) -> bytes:
        """Decompress data using specified compression type."""
        if compression == CompressionType.GZIP:
            return gzip.decompress(data)

        elif compression == CompressionType.LZMA:
            return lzma.decompress(data)

        else:
            return data

    def _update_serialization_stats(self, format: SerializationFormat, processing_time: float, success: bool):
        """Update serialization statistics."""
        self.serialization_stats["total_serialized"] += 1
        self.serialization_stats["serialization_time"] += processing_time

        if not success:
            self.serialization_stats["errors"] += 1

        # Update by format
        format_key = format.value
        if format_key not in self.serialization_stats["by_format"]:
            self.serialization_stats["by_format"][format_key] = {
                "count": 0,
                "total_time": 0.0,
                "errors": 0
            }

        self.serialization_stats["by_format"][format_key]["count"] += 1
        self.serialization_stats["by_format"][format_key]["total_time"] += processing_time
        if not success:
            self.serialization_stats["by_format"][format_key]["errors"] += 1

    def _update_deserialization_stats(self, format: SerializationFormat, processing_time: float, success: bool):
        """Update deserialization statistics."""
        self.serialization_stats["total_deserialized"] += 1
        self.serialization_stats["deserialization_time"] += processing_time

        if not success:
            self.serialization_stats["errors"] += 1

        # Update by format
        format_key = format.value
        if format_key not in self.serialization_stats["by_format"]:
            self.serialization_stats["by_format"][format_key] = {
                "count": 0,
                "total_time": 0.0,
                "errors": 0
            }

        self.serialization_stats["by_format"][format_key]["count"] += 1
        self.serialization_stats["by_format"][format_key]["total_time"] += processing_time
        if not success:
            self.serialization_stats["by_format"][format_key]["errors"] += 1

    # Batch operations
    async def serialize_batch(self, messages: List[RichMessage],
                            format: Optional[SerializationFormat] = None) -> List[SerializedMessage]:
        """Serialize multiple messages."""
        results = []
        for message in messages:
            try:
                serialized = await self.serialize(message, format)
                results.append(serialized)
            except Exception as e:
                self.logger.error(f"Batch serialization failed for message {message.id}: {str(e)}")
                # Could add error handling or continue with next message
        return results

    async def deserialize_batch(self, serialized_messages: List[SerializedMessage]) -> List[RichMessage]:
        """Deserialize multiple messages."""
        results = []
        for serialized_msg in serialized_messages:
            try:
                message = await self.deserialize(serialized_msg)
                results.append(message)
            except Exception as e:
                self.logger.error(f"Batch deserialization failed: {str(e)}")
                # Could add error handling or continue with next message
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive serialization statistics."""
        stats = self.serialization_stats.copy()

        # Calculate averages
        if stats["total_serialized"] > 0:
            stats["average_serialization_time"] = stats["serialization_time"] / stats["total_serialized"]
        else:
            stats["average_serialization_time"] = 0.0

        if stats["total_deserialized"] > 0:
            stats["average_deserialization_time"] = stats["deserialization_time"] / stats["total_deserialized"]
        else:
            stats["average_deserialization_time"] = 0.0

        # Calculate error rates
        total_operations = stats["total_serialized"] + stats["total_deserialized"]
        if total_operations > 0:
            stats["error_rate"] = stats["errors"] / total_operations
        else:
            stats["error_rate"] = 0.0

        # Compression statistics
        if stats["compression_saves"] > 0:
            stats["average_compression_save"] = stats["total_bytes_saved"] / stats["compression_saves"]
        else:
            stats["average_compression_save"] = 0.0

        # Format-specific statistics
        for format_key, format_stats in stats["by_format"].items():
            if format_stats["count"] > 0:
                format_stats["average_time"] = format_stats["total_time"] / format_stats["count"]
                format_stats["error_rate"] = format_stats["errors"] / format_stats["count"]
            else:
                format_stats["average_time"] = 0.0
                format_stats["error_rate"] = 0.0

        return stats

    def reset_stats(self):
        """Reset serialization statistics."""
        self.serialization_stats = {
            "total_serialized": 0,
            "total_deserialized": 0,
            "serialization_time": 0.0,
            "deserialization_time": 0.0,
            "compression_saves": 0,
            "total_bytes_saved": 0,
            "by_format": {},
            "errors": 0
        }


class MessagePersistence:
    """Message persistence with multiple backend support."""

    def __init__(self, backend: PersistenceBackend = PersistenceBackend.FILE,
                 config: Dict[str, Any] = None):
        """Initialize message persistence."""
        self.backend = backend
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize serializer
        serializer_config = self.config.get("serializer", {})
        self.serializer = MessageSerializer(serializer_config)

        # Initialize backend
        self._initialize_backend()

        # Persistence statistics
        self.persistence_stats = {
            "total_saved": 0,
            "total_loaded": 0,
            "save_time": 0.0,
            "load_time": 0.0,
            "errors": 0
        }

    def _initialize_backend(self):
        """Initialize persistence backend."""
        if self.backend == PersistenceBackend.FILE:
            self.storage_dir = Path(self.config.get("storage_dir", "./message_storage"))
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        elif self.backend == PersistenceBackend.SQLITE:
            self.db_path = self.config.get("db_path", "./messages.db")
            self._initialize_sqlite()

    def _initialize_sqlite(self):
        """Initialize SQLite database."""
        self.db_connection = sqlite3.connect(self.db_path)
        self.db_connection.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create SQLite tables."""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                data BLOB NOT NULL,
                format TEXT NOT NULL,
                compression TEXT NOT NULL,
                checksum TEXT,
                version TEXT,
                timestamp TEXT,
                original_size INTEGER,
                compressed_size INTEGER,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_message_id ON messages(message_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_message_type ON messages(message_type)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)
        ''')

        self.db_connection.commit()

    async def save_message(self, message: RichMessage,
                          filename: Optional[str] = None,
                          format: Optional[SerializationFormat] = None) -> bool:
        """Save a message to persistence backend."""
        start_time = datetime.now()

        try:
            # Serialize message
            serialized = await self.serializer.serialize(message, format)

            if self.backend == PersistenceBackend.FILE:
                success = await self._save_to_file(serialized, filename)
            elif self.backend == PersistenceBackend.SQLITE:
                success = await self._save_to_sqlite(serialized)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")

            # Update statistics
            if success:
                processing_time = (datetime.now() - start_time).total_seconds()
                self.persistence_stats["total_saved"] += 1
                self.persistence_stats["save_time"] += processing_time

            return success

        except Exception as e:
            self.logger.error(f"Failed to save message {message.id}: {str(e)}")
            self.persistence_stats["errors"] += 1
            return False

    async def load_message(self, message_id: str,
                          filename: Optional[str] = None) -> Optional[RichMessage]:
        """Load a message from persistence backend."""
        start_time = datetime.now()

        try:
            if self.backend == PersistenceBackend.FILE:
                serialized = await self._load_from_file(message_id, filename)
            elif self.backend == PersistenceBackend.SQLITE:
                serialized = await self._load_from_sqlite(message_id)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")

            if serialized is None:
                return None

            # Deserialize message
            message = await self.serializer.deserialize(serialized)

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.persistence_stats["total_loaded"] += 1
            self.persistence_stats["load_time"] += processing_time

            return message

        except Exception as e:
            self.logger.error(f"Failed to load message {message_id}: {str(e)}")
            self.persistence_stats["errors"] += 1
            return None

    async def _save_to_file(self, serialized: SerializedMessage, filename: Optional[str] None) -> bool:
        """Save serialized message to file."""
        if filename is None:
            # Generate filename from message metadata
            timestamp = serialized.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{serialized.message_id}_{timestamp}.msg"

        file_path = self.storage_dir / filename

        try:
            # Write serialized data and metadata
            with open(file_path, 'wb') as f:
                # Write header with metadata
                header = {
                    "format": serialized.format.value,
                    "compression": serialized.compression.value,
                    "original_size": serialized.original_size,
                    "checksum": serialized.checksum,
                    "version": serialized.version,
                    "timestamp": serialized.timestamp.isoformat(),
                    "message_id": serialized.message_id,
                    "message_type": serialized.message_type
                }

                header_bytes = json.dumps(header).encode('utf-8')
                header_size = len(header_bytes)

                # Write header size and header
                f.write(header_size.to_bytes(4, 'big'))
                f.write(header_bytes)

                # Write message data
                f.write(serialized.data)

            return True

        except Exception as e:
            self.logger.error(f"Failed to save to file {file_path}: {str(e)}")
            return False

    async def _load_from_file(self, message_id: str, filename: Optional[str] = None) -> Optional[SerializedMessage]:
        """Load serialized message from file."""
        if filename is None:
            # Find file by message ID
            pattern = f"{message_id}_*.msg"
            matching_files = list(self.storage_dir.glob(pattern))
            if not matching_files:
                return None
            # Use the most recent file
            file_path = max(matching_files, key=lambda p: p.stat().st_mtime)
        else:
            file_path = self.storage_dir / filename

        try:
            with open(file_path, 'rb') as f:
                # Read header size
                header_size_bytes = f.read(4)
                header_size = int.from_bytes(header_size_bytes, 'big')

                # Read header
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes.decode('utf-8'))

                # Read message data
                data = f.read()

                # Create SerializedMessage object
                return SerializedMessage(
                    data=data,
                    format=SerializationFormat(header["format"]),
                    compression=CompressionType(header["compression"]),
                    original_size=header["original_size"],
                    compressed_size=len(data),
                    checksum=header.get("checksum", ""),
                    version=header.get("version", "1.0"),
                    timestamp=datetime.fromisoformat(header["timestamp"]),
                    message_id=header["message_id"],
                    message_type=header["message_type"]
                )

        except Exception as e:
            self.logger.error(f"Failed to load from file {file_path}: {str(e)}")
            return None

    async def _save_to_sqlite(self, serialized: SerializedMessage) -> bool:
        """Save serialized message to SQLite database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO messages
                (id, message_id, message_type, data, format, compression, checksum,
                 version, timestamp, original_size, compressed_size, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"{serialized.message_id}_{serialized.timestamp.isoformat()}",
                serialized.message_id,
                serialized.message_type,
                serialized.data,
                serialized.format.value,
                serialized.compression.value,
                serialized.checksum,
                serialized.version,
                serialized.timestamp.isoformat(),
                serialized.original_size,
                serialized.compressed_size,
                json.dumps(serialized.metadata)
            ))

            self.db_connection.commit()
            return True

        except Exception as e:
            self.logger.error(f"Failed to save to SQLite: {str(e)}")
            return False

    async def _load_from_sqlite(self, message_id: str) -> Optional[SerializedMessage]:
        """Load serialized message from SQLite database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT * FROM messages WHERE message_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (message_id,))

            row = cursor.fetchone()
            if row is None:
                return None

            return SerializedMessage(
                data=row['data'],
                format=SerializationFormat(row['format']),
                compression=CompressionType(row['compression']),
                original_size=row['original_size'],
                compressed_size=row['compressed_size'],
                checksum=row['checksum'] or "",
                version=row['version'] or "1.0",
                timestamp=datetime.fromisoformat(row['timestamp']),
                message_id=row['message_id'],
                message_type=row['message_type'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )

        except Exception as e:
            self.logger.error(f"Failed to load from SQLite: {str(e)}")
            return None

    # Batch operations
    async def save_batch(self, messages: List[RichMessage]) -> List[bool]:
        """Save multiple messages."""
        results = []
        for message in messages:
            result = await self.save_message(message)
            results.append(result)
        return results

    async def load_batch(self, message_ids: List[str]) -> List[Optional[RichMessage]]:
        """Load multiple messages."""
        results = []
        for message_id in message_ids:
            message = await self.load_message(message_id)
            results.append(message)
        return results

    def get_persistence_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        stats = self.persistence_stats.copy()

        # Calculate averages
        if stats["total_saved"] > 0:
            stats["average_save_time"] = stats["save_time"] / stats["total_saved"]
        else:
            stats["average_save_time"] = 0.0

        if stats["total_loaded"] > 0:
            stats["average_load_time"] = stats["load_time"] / stats["total_loaded"]
        else:
            stats["average_load_time"] = 0.0

        # Calculate error rate
        total_operations = stats["total_saved"] + stats["total_loaded"]
        if total_operations > 0:
            stats["error_rate"] = stats["errors"] / total_operations
        else:
            stats["error_rate"] = 0.0

        return stats

    def reset_stats(self):
        """Reset persistence statistics."""
        self.persistence_stats = {
            "total_saved": 0,
            "total_loaded": 0,
            "save_time": 0.0,
            "load_time": 0.0,
            "errors": 0
        }

    def close(self):
        """Close persistence resources."""
        if self.backend == PersistenceBackend.SQLITE:
            self.db_connection.close()