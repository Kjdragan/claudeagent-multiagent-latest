"""
Message Processing Utilities - Comprehensive Rich Message Processing and Display System

This package provides advanced message processing capabilities for the multi-agent research system,
including type-specific handlers, rich formatting, content analysis, routing, and optimization.

Key Features:
- Comprehensive message type system with type-specific processing
- Rich display formatting with proper visualization
- Message analysis and content enhancement capabilities
- Message routing and filtering based on content and context
- Message caching and optimization for performance
- Integration with enhanced orchestrator and sub-agent systems
- Comprehensive error handling and validation

Components:
- Core: Base message types and processing infrastructure
- Formatters: Rich display formatting and visualization
- Analyzers: Message analysis and content enhancement
- Routers: Message routing and filtering system
- Cache: Message caching and optimization
- Display: Rich display patterns and visualization
- Serializers: Message serialization and persistence
- Tests: Comprehensive testing and validation
"""

from .core.message_types import *
from .core.message_processor import *
from .core.message_analyzer import *
from .formatters.rich_formatter import *
from .analyzers.content_enhancer import *
from .routers.message_router import *
from .cache.message_cache import *
from .display.display_manager import *

__version__ = "2.3.0"
__author__ = "Multi-Agent Research System"
__description__ = "Comprehensive rich message processing and display system"

__all__ = [
    # Core message types and infrastructure
    "EnhancedMessageType",
    "RichMessage",
    "MessagePriority",
    "MessageContext",

    # Core processing
    "MessageProcessor",
    "MessageAnalyzer",

    # Formatting and display
    "RichFormatter",
    "DisplayManager",

    # Analysis and enhancement
    "ContentEnhancer",
    "MessageQualityAnalyzer",

    # Routing and filtering
    "MessageRouter",
    "MessageFilter",

    # Caching and optimization
    "MessageCache",
    "MessageOptimizer",

    # Serialization
    "MessageSerializer",
]

# Package-level configuration
DEFAULT_CONFIG = {
    "max_message_history": 1000,
    "cache_size": 500,
    "enable_analysis": True,
    "enable_enhancement": True,
    "enable_caching": True,
    "default_format": "rich",
    "performance_tracking": True
}

def get_message_processor(config=None):
    """Factory function to get configured message processor."""
    from .core.message_processor import MessageProcessor

    if config is None:
        config = DEFAULT_CONFIG.copy()

    return MessageProcessor(config)

def get_rich_formatter(config=None):
    """Factory function to get configured rich formatter."""
    from .formatters.rich_formatter import RichFormatter

    if config is None:
        config = DEFAULT_CONFIG.copy()

    return RichFormatter(config)

def get_message_router(config=None):
    """Factory function to get configured message router."""
    from .routers.message_router import MessageRouter

    if config is None:
        config = DEFAULT_CONFIG.copy()

    return MessageRouter(config)