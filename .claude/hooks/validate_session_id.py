#!/usr/bin/env python3
"""
Session ID Validation Hook

This hook validates session IDs to ensure they follow the expected UUID format
and prevents the creation of date-based session IDs that cause file organization issues.
"""

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def validate_session_id(session_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Validate session ID format and prevent problematic session IDs.

    Args:
        session_id: The session ID to validate
        context: Additional context about the session creation

    Returns:
        Dictionary with validation result and any corrections needed
    """

    if not session_id:
        return {
            "valid": False,
            "error": "Session ID cannot be empty",
            "corrected_id": str(uuid.uuid4()),
            "reason": "Generated new UUID session ID"
        }

    # Check if it's a valid UUID
    try:
        uuid.UUID(session_id)
        return {
            "valid": True,
            "session_id": session_id,
            "format": "uuid",
            "reason": "Valid UUID session ID"
        }
    except ValueError:
        pass

    # Check for problematic date-based patterns
    date_patterns = [
        r'\d{4}_\d{2}_\d{2}',  # 2024_10_14
        r'\d{4}-\d{2}-\d{2}',  # 2024-10-14
        r'\d{8}',            # 20241014
    ]

    for pattern in date_patterns:
        if re.search(pattern, session_id):
            # This looks like a date-based session ID, which is problematic
            corrected_id = str(uuid.uuid4())

            return {
                "valid": False,
                "error": f"Date-based session ID detected: {session_id}",
                "corrected_id": corrected_id,
                "pattern_matched": pattern,
                "reason": "Date-based session IDs cause file organization issues",
                "suggestion": "Use UUID format for consistent session management"
            }

    # Check for other problematic patterns
    problematic_patterns = [
        r'.*_research_\d{4}_\d{2}_\d{2}.*',  # topic_research_2024_10_14
        r'.*_\d{4}_\d{2}_\d{4}.*',        # topic_2024_10_1415
    ]

    for pattern in problematic_patterns:
        if re.search(pattern, session_id):
            corrected_id = str(uuid.uuid4())

            return {
                "valid": False,
                "error": f"Problematic session ID pattern detected: {session_id}",
                "corrected_id": corrected_id,
                "pattern_matched": pattern,
                "reason": "Session ID contains date components that may cause conflicts",
                "suggestion": "Use clean UUID format without date components"
            }

    # Check for reasonable length limits
    if len(session_id) > 100:
        corrected_id = str(uuid.uuid4())

        return {
            "valid": False,
            "error": f"Session ID too long ({len(session_id)} characters): {session_id}",
            "corrected_id": corrected_id,
            "reason": "Session IDs should be concise for file system compatibility",
            "suggestion": "Use UUID format for optimal compatibility"
        }

    # Check for file system incompatible characters
    invalid_chars = r'[<>:"/\\|?*]'
    if re.search(invalid_chars, session_id):
        corrected_id = str(uuid.uuid4())

        return {
            "valid": False,
            "error": f"Session ID contains invalid file system characters: {session_id}",
            "corrected_id": corrected_id,
            "invalid_chars": re.findall(invalid_chars, session_id),
            "reason": "Session IDs must be file system compatible",
            "suggestion": "Use UUID format for maximum compatibility"
        }

    # If we get here, the session ID passed basic validation but isn't a UUID
    # Generate a UUID recommendation
    return {
        "valid": True,
        "session_id": session_id,
        "format": "custom",
        "warning": "Session ID is not in UUID format",
        "suggested_uuid": str(uuid.uuid4()),
        "reason": "Consider using UUID format for consistency",
        "suggestion": "UUID format provides better uniqueness and compatibility"
    }


def ensure_session_id_format(session_id: str, context: Dict[str, Any] = None) -> str:
    """
    Ensure session ID is in the correct format, generating a new one if needed.

    Args:
        session_id: The session ID to validate/format
        context: Additional context about the session creation

    Returns:
        A valid session ID (original if valid, new one if not)
    """

    validation_result = validate_session_id(session_id, context)

    if validation_result["valid"]:
        return validation_result["session_id"]
    else:
        # Return the corrected ID
        if "corrected_id" in validation_result:
            return validation_result["corrected_id"]
        else:
            # Generate a new UUID as fallback
            return str(uuid.uuid4())


def validate_session_directory_structure(session_id: str, base_path: str = "KEVIN/sessions") -> Dict[str, Any]:
    """
    Validate that the session directory structure is correct for the given session ID.

    Args:
        session_id: The session ID to validate
        base_path: Base path for sessions

    Returns:
        Dictionary with validation result
    """

    session_path = Path(base_path) / session_id

    if not session_path.exists():
        return {
            "valid": True,
            "session_path": str(session_path),
            "exists": False,
            "reason": "Session directory does not exist yet (expected for new sessions)"
        }

    # Check for expected subdirectories
    expected_subdirs = ["working", "research", "agent_logs"]
    missing_subdirs = []
    existing_subdirs = []

    for subdir in expected_subdirs:
        subdir_path = session_path / subdir
        if subdir_path.exists():
            existing_subdirs.append(subdir)
        else:
            missing_subdirs.append(subdir)

    # Check for problematic date-based subdirectories that might indicate
    # incorrect session ID usage
    all_subdirs = [d.name for d in session_path.iterdir() if d.is_dir()]
    date_based_subdirs = []

    for subdir in all_subdirs:
        date_patterns = [r'\d{4}_\d{2}_\d{2}', r'\d{4}-\d{2}-\d{2}']
        for pattern in date_patterns:
            if re.search(pattern, subdir):
                date_based_subdirs.append(subdir)
                break

    return {
        "valid": len(date_based_subdirs) == 0,
        "session_path": str(session_path),
        "exists": True,
        "expected_subdirs": {
            "existing": existing_subdirs,
            "missing": missing_subdirs
        },
        "date_based_subdirs": date_based_subdirs,
        "all_subdirs": all_subdirs,
        "issues": len(date_based_subdirs) > 0,
        "reason": "Session directory structure validation" if len(date_based_subdirs) == 0 else "Date-based subdirectories detected"
    }


# Hook interface for Claude Agent SDK
def pre_session_creation(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook called before session creation to validate session ID.

    Args:
        context: Context dictionary containing session creation parameters

    Returns:
        Updated context with validated session ID
    """

    session_id = context.get("session_id")
    if not session_id:
        # Generate a new UUID if no session ID provided
        session_id = str(uuid.uuid4())
        context["session_id"] = session_id

        return {
            "context": context,
            "action_taken": "generated_new_session_id",
            "session_id": session_id,
            "reason": "No session ID provided, generated UUID"
        }

    # Validate existing session ID
    validation_result = validate_session_id(session_id, context)

    if not validation_result["valid"]:
        # Use corrected session ID
        corrected_id = validation_result["corrected_id"]
        context["session_id"] = corrected_id

        return {
            "context": context,
            "action_taken": "corrected_session_id",
            "original_session_id": session_id,
            "new_session_id": corrected_id,
            "validation_result": validation_result,
            "reason": "Invalid session ID corrected"
        }

    # Session ID is valid
    return {
        "context": context,
        "action_taken": "validated_session_id",
        "session_id": session_id,
        "validation_result": validation_result,
        "reason": "Session ID validated successfully"
    }


def post_session_creation(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook called after session creation to validate directory structure.

    Args:
        context: Context dictionary containing session creation information

    Returns:
        Validation result for session directory structure
    """

    session_id = context.get("session_id")
    if not session_id:
        return {
            "valid": False,
            "error": "No session ID in context",
            "reason": "Cannot validate without session ID"
        }

    # Validate directory structure
    directory_validation = validate_session_directory_structure(session_id)

    return {
        "session_id": session_id,
        "directory_validation": directory_validation,
        "reason": "Post-creation directory structure validation"
    }


# Export hook functions
__all__ = [
    "validate_session_id",
    "ensure_session_id_format",
    "validate_session_directory_structure",
    "pre_session_creation",
    "post_session_creation"
]