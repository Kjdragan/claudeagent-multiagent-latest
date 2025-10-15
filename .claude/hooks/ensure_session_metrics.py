#!/usr/bin/env python3
"""
Session Metrics Validation Hook

This hook ensures that session metrics are properly initialized and maintained
throughout the research workflow to prevent key errors.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


def validate_session_metrics_structure(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that session metadata has the required session_metrics structure.

    Args:
        metadata: Session metadata dictionary

    Returns:
        Dictionary with validation result and any corrections needed
    """

    validation_result = {
        "valid": True,
        "issues": [],
        "corrections": [],
        "missing_keys": [],
        "invalid_types": []
    }

    # Check if session_metrics exists
    if "session_metrics" not in metadata:
        validation_result["valid"] = False
        validation_result["issues"].append({
            "type": "missing_key",
            "key": "session_metrics",
            "message": "session_metrics key is missing from metadata"
        })
        validation_result["missing_keys"].append("session_metrics")

        # Add correction
        default_metrics = get_default_session_metrics()
        validation_result["corrections"].append({
            "type": "add_key",
            "key": "session_metrics",
            "value": default_metrics,
            "message": "Added default session_metrics structure"
        })

    else:
        # Validate session_metrics structure
        session_metrics = metadata["session_metrics"]

        if not isinstance(session_metrics, dict):
            validation_result["valid"] = False
            validation_result["issues"].append({
                "type": "invalid_type",
                "key": "session_metrics",
                "expected_type": "dict",
                "actual_type": type(session_metrics).__name__,
                "message": "session_metrics must be a dictionary"
            })
            validation_result["invalid_types"].append("session_metrics")

            # Add correction
            default_metrics = get_default_session_metrics()
            validation_result["corrections"].append({
                "type": "replace_value",
                "key": "session_metrics",
                "value": default_metrics,
                "message": "Replaced invalid session_metrics with default structure"
            })
        else:
            # Check for required keys in session_metrics
            required_keys = [
                "duration_seconds",
                "total_urls_processed",
                "successful_scrapes",
                "quality_score",
                "completion_percentage"
            ]

            for key in required_keys:
                if key not in session_metrics:
                    validation_result["issues"].append({
                        "type": "missing_key",
                        "key": f"session_metrics.{key}",
                        "message": f"Required key '{key}' missing from session_metrics"
                    })
                    validation_result["missing_keys"].append(f"session_metrics.{key}")

                    # Add default value
                    default_value = get_default_session_metrics().get(key, 0)
                    validation_result["corrections"].append({
                        "type": "add_key",
                        "key": f"session_metrics.{key}",
                        "value": default_value,
                        "message": f"Added default value for session_metrics.{key}"
                    })

            # Validate data types for existing keys
            type_requirements = {
                "duration_seconds": (int, float),
                "total_urls_processed": int,
                "successful_scrapes": int,
                "quality_score": (int, float),
                "completion_percentage": int
            }

            for key, expected_types in type_requirements.items():
                if key in session_metrics:
                    current_value = session_metrics[key]
                    if not isinstance(current_value, expected_types):
                        validation_result["issues"].append({
                            "type": "invalid_type",
                            "key": f"session_metrics.{key}",
                            "expected_type": str(expected_types),
                            "actual_type": type(current_value).__name__,
                            "message": f"Invalid type for session_metrics.{key}"
                        })
                        validation_result["invalid_types"].append(f"session_metrics.{key}")

                        # Add type correction
                        default_value = get_default_session_metrics().get(key, 0)
                        validation_result["corrections"].append({
                            "type": "fix_type",
                            "key": f"session_metrics.{key}",
                            "value": default_value,
                            "message": f"Fixed type for session_metrics.{key}"
                        })

    return validation_result


def get_default_session_metrics() -> Dict[str, Any]:
    """
    Get the default session metrics structure.

    Returns:
        Default session metrics dictionary
    """

    return {
        "duration_seconds": 0,
        "total_urls_processed": 0,
        "successful_scrapes": 0,
        "quality_score": None,
        "completion_percentage": 0,
        "final_report_generated": False,
        "stage_completion_times": {},
        "error_count": 0,
        "retry_count": 0,
        "resource_usage": {
            "memory_peak_mb": 0,
            "cpu_peak_percent": 0
        }
    }


def ensure_session_metrics(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure session metadata has a valid session_metrics structure.

    Args:
        metadata: Session metadata dictionary

    Returns:
        Updated metadata with valid session_metrics
    """

    validation_result = validate_session_metrics_structure(metadata)

    # Apply corrections if needed
    for correction in validation_result["corrections"]:
        if correction["type"] == "add_key":
            if correction["key"] == "session_metrics":
                metadata["session_metrics"] = correction["value"]
            elif correction["key"].startswith("session_metrics."):
                key_name = correction["key"].replace("session_metrics.", "")
                metadata["session_metrics"][key_name] = correction["value"]

        elif correction["type"] == "replace_value":
            metadata["session_metrics"] = correction["value"]

        elif correction["type"] == "fix_type":
            key_name = correction["key"].replace("session_metrics.", "")
            metadata["session_metrics"][key_name] = correction["value"]

    return metadata


def load_and_validate_session_metadata(session_id: str,
                                     base_path: str = "KEVIN/sessions") -> Dict[str, Any]:
    """
    Load and validate session metadata from file.

    Args:
        session_id: Session ID to load
        base_path: Base path for sessions

    Returns:
        Dictionary with loaded metadata and validation result
    """

    metadata_path = Path(base_path) / session_id / "session_metadata.json"

    if not metadata_path.exists():
        return {
            "session_id": session_id,
            "exists": False,
            "error": f"Session metadata file not found: {metadata_path}",
            "metadata": None,
            "validation": None
        }

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        return {
            "session_id": session_id,
            "exists": True,
            "error": f"Failed to load session metadata: {e}",
            "metadata": None,
            "validation": None
        }

    # Validate session metrics
    validation_result = validate_session_metrics_structure(metadata)

    return {
        "session_id": session_id,
        "exists": True,
        "metadata": metadata,
        "validation": validation_result,
        "needs_correction": not validation_result["valid"]
    }


def save_session_metadata_with_validation(session_id: str,
                                        metadata: Dict[str, Any],
                                        base_path: str = "KEVIN/sessions") -> Dict[str, Any]:
    """
    Save session metadata after validating session_metrics.

    Args:
        session_id: Session ID
        metadata: Session metadata to save
        base_path: Base path for sessions

    Returns:
        Dictionary with save result
    """

    # Ensure session_metrics is valid
    metadata = ensure_session_metrics(metadata)

    metadata_path = Path(base_path) / session_id / "session_metadata.json"

    try:
        # Ensure directory exists
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return {
            "session_id": session_id,
            "saved": True,
            "metadata_path": str(metadata_path),
            "validation": "session_metrics ensured before saving"
        }

    except Exception as e:
        return {
            "session_id": session_id,
            "saved": False,
            "error": f"Failed to save session metadata: {e}",
            "metadata_path": str(metadata_path)
        }


def update_session_metrics(session_id: str,
                          updates: Dict[str, Any],
                          base_path: str = "KEVIN/sessions") -> Dict[str, Any]:
    """
    Update session metrics for a session.

    Args:
        session_id: Session ID to update
        updates: Dictionary of updates to apply
        base_path: Base path for sessions

    Returns:
        Dictionary with update result
    """

    # Load existing metadata
    load_result = load_and_validate_session_metadata(session_id, base_path)

    if not load_result["metadata"]:
        return {
            "session_id": session_id,
            "updated": False,
            "error": "Could not load existing metadata",
            "load_result": load_result
        }

    metadata = load_result["metadata"]

    # Ensure session_metrics exists
    metadata = ensure_session_metrics(metadata)

    # Apply updates
    for key, value in updates.items():
        if key.startswith("session_metrics."):
            # Update nested session_metrics
            nested_key = key.replace("session_metrics.", "")
            metadata["session_metrics"][nested_key] = value
        else:
            # Update top-level metadata
            metadata[key] = value

    # Save updated metadata
    save_result = save_session_metadata_with_validation(session_id, metadata, base_path)

    return {
        "session_id": session_id,
        "updated": save_result["saved"],
        "updates_applied": updates,
        "save_result": save_result
    }


def validate_session_metrics_across_sessions(base_path: str = "KEVIN/sessions") -> Dict[str, Any]:
    """
    Validate session metrics across all sessions.

    Args:
        base_path: Base path for sessions

    Returns:
        Dictionary with validation results for all sessions
    """

    sessions_path = Path(base_path)

    if not sessions_path.exists():
        return {
            "base_path": base_path,
            "exists": False,
            "error": f"Sessions directory not found: {sessions_path}"
        }

    validation_results = {}
    total_sessions = 0
    valid_sessions = 0
    sessions_needing_correction = 0

    for session_dir in sessions_path.iterdir():
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name
        total_sessions += 1

        # Validate session metrics
        validation_result = load_and_validate_session_metadata(session_id, base_path)
        validation_results[session_id] = validation_result

        if validation_result["metadata"]:
            if validation_result["validation"]["valid"]:
                valid_sessions += 1
            else:
                sessions_needing_correction += 1

    return {
        "base_path": base_path,
        "total_sessions": total_sessions,
        "valid_sessions": valid_sessions,
        "sessions_needing_correction": sessions_needing_correction,
        "validation_results": validation_results
    }


# Hook interface for Claude Agent SDK
def pre_session_update(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook called before session metadata update to ensure session_metrics.

    Args:
        context: Context dictionary containing session update information

    Returns:
        Updated context with ensured session_metrics
    """

    session_id = context.get("session_id")
    metadata = context.get("metadata")
    updates = context.get("updates", {})

    if not session_id or not metadata:
        return {
            "valid": False,
            "error": "Missing session_id or metadata in context",
            "reason": "Cannot validate without session_id and metadata"
        }

    # Ensure session_metrics exists
    updated_metadata = ensure_session_metrics(metadata)

    # Validate the updates don't break session_metrics structure
    if any(key.startswith("session_metrics.") for key in updates.keys()):
        # Apply updates to a copy to validate
        test_metadata = updated_metadata.copy()
        for key, value in updates.items():
            if key.startswith("session_metrics."):
                nested_key = key.replace("session_metrics.", "")
                test_metadata["session_metrics"][nested_key] = value

        validation_result = validate_session_metrics_structure(test_metadata)

        if not validation_result["valid"]:
            return {
                "valid": False,
                "error": "Session metrics update would break structure",
                "validation_result": validation_result,
                "updates": updates,
                "reason": "Session metrics update validation failed"
            }

    return {
        "valid": True,
        "context": {
            **context,
            "metadata": updated_metadata
        },
        "validation_result": validate_session_metrics_structure(updated_metadata),
        "reason": "Session metrics ensured before update"
    }


def post_session_update(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook called after session metadata update to validate session_metrics.

    Args:
        context: Context dictionary containing session update information

    Returns:
        Validation result for updated session metrics
    """

    session_id = context.get("session_id")
    metadata = context.get("metadata")

    if not session_id or not metadata:
        return {
            "valid": False,
            "error": "Missing session_id or metadata in context",
            "reason": "Cannot validate without session_id and metadata"
        }

    validation_result = validate_session_metrics_structure(metadata)

    return {
        "session_id": session_id,
        "validation_result": validation_result,
        "reason": "Post-update session metrics validation"
    }


# Export hook functions
__all__ = [
    "validate_session_metrics_structure",
    "get_default_session_metrics",
    "ensure_session_metrics",
    "load_and_validate_session_metadata",
    "save_session_metadata_with_validation",
    "update_session_metrics",
    "validate_session_metrics_across_sessions",
    "pre_session_update",
    "post_session_update"
]