"""
Debug Checkpoint System

Saves intermediate state and data for debugging and inspection.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class DebugCheckpoint:
    """Save intermediate state for debugging."""
    
    def __init__(self, session_id: str):
        """Initialize debug checkpoint manager.
        
        Args:
            session_id: Session identifier
        """
        self.session_id = session_id
        self.session_dir = Path.home() / "lrepos" / "claudeagent-multiagent-latest" / "KEVIN" / "sessions" / session_id
        self.checkpoint_dir = self.session_dir / "debug_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, name: str, data: Dict[str, Any]):
        """Save checkpoint with timestamp.
        
        Args:
            name: Checkpoint name (e.g., 'educational_context', 'report_prompt')
            data: Dictionary of data to save
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.json"
            filepath = self.checkpoint_dir / filename
            
            # Add metadata
            checkpoint_data = {
                "checkpoint_name": name,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.debug(f"📌 Saved checkpoint: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {name}: {e}")
    
    def save_text(self, name: str, content: str):
        """Save text content with timestamp.
        
        Args:
            name: Checkpoint name
            content: Text content to save
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.txt"
            filepath = self.checkpoint_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Checkpoint: {name}\n")
                f.write(f"# Session: {self.session_id}\n")
                f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"# {'='*76}\n\n")
                f.write(content)
            
            logger.debug(f"📌 Saved text checkpoint: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save text checkpoint {name}: {e}")
