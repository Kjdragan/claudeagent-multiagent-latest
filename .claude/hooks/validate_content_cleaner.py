#!/usr/bin/env python3
"""
Content Cleaner Validation Hook

This hook validates content cleaner method calls and ensures the correct
methods are used for the ModernWebContentCleaner class.
"""

import inspect
from typing import Any, Dict, Optional, Union
from pathlib import Path


def validate_modern_web_content_cleaner_usage(content_cleaner_instance: Any,
                                             method_name: str,
                                             args: tuple = (),
                                             kwargs: dict = None) -> Dict[str, Any]:
    """
    Validate that the correct method is being called on ModernWebContentCleaner.

    Args:
        content_cleaner_instance: The ModernWebContentCleaner instance
        method_name: The method name being called
        args: Positional arguments being passed
        kwargs: Keyword arguments being passed

    Returns:
        Dictionary with validation result and corrections if needed
    """

    if kwargs is None:
        kwargs = {}

    # Get the class name
    class_name = content_cleaner_instance.__class__.__name__

    # Check if this is the ModernWebContentCleaner
    if class_name != "ModernWebContentCleaner":
        return {
            "valid": True,
            "class_name": class_name,
            "method_name": method_name,
            "reason": "Not a ModernWebContentCleaner instance"
        }

    # Get available methods on ModernWebContentCleaner
    available_methods = []
    for name, method in inspect.getmembers(content_cleaner_instance, predicate=inspect.ismethod):
        if not name.startswith('_'):
            available_methods.append(name)

    # Get available functions (module-level functions)
    import multi_agent_research_system.utils.modern_content_cleaner as cleaner_module
    for name in dir(cleaner_module):
        if callable(getattr(cleaner_module, name)) and not name.startswith('_'):
            available_methods.append(name)

    # Check for common incorrect method calls
    incorrect_method_corrections = {
        "clean_content": "clean_article_content",
        "clean": "clean_article_content",
        "process_content": "clean_article_content",
        "parse_content": "clean_article_content"
    }

    validation_result = {
        "class_name": class_name,
        "method_name": method_name,
        "available_methods": available_methods,
        "valid": False,
        "corrections": [],
        "warnings": []
    }

    # Check if the method exists
    if method_name not in available_methods:
        # Check for common incorrect method names
        if method_name in incorrect_method_corrections:
            correct_method = incorrect_method_corrections[method_name]
            validation_result["valid"] = False
            validation_result["corrections"].append({
                "type": "incorrect_method_name",
                "incorrect_method": method_name,
                "correct_method": correct_method,
                "message": f"Method '{method_name}' does not exist. Use '{correct_method}' instead.",
                "fixed_call": f"content_cleaner.{correct_method}(*args, **kwargs)"
            })
        else:
            # Method doesn't exist, suggest available methods
            similar_methods = [m for m in available_methods if method_name.lower() in m.lower()]

            validation_result["valid"] = False
            validation_result["corrections"].append({
                "type": "method_not_found",
                "requested_method": method_name,
                "available_methods": available_methods,
                "similar_methods": similar_methods,
                "message": f"Method '{method_name}' not found in ModernWebContentCleaner.",
                "suggestions": similar_methods if similar_methods else available_methods[:5]
            })
    else:
        # Method exists, validate its usage
        validation_result["valid"] = True

        # Validate arguments for specific methods
        if method_name == "clean_article_content":
            arg_validation = _validate_clean_article_content_args(args, kwargs)
            validation_result.update(arg_validation)
        elif method_name == "apply_modern_cleaning":
            arg_validation = _validate_apply_modern_cleaning_args(args, kwargs)
            validation_result.update(arg_validation)

    return validation_result


def _validate_clean_article_content_args(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Validate arguments for clean_article_content method."""

    validation = {
        "arg_validation": {
            "valid": True,
            "issues": [],
            "warnings": []
        }
    }

    # Check required arguments
    if len(args) < 1 and "content" not in kwargs:
        validation["arg_validation"]["valid"] = False
        validation["arg_validation"]["issues"].append({
            "type": "missing_required_argument",
            "argument": "content",
            "message": "clean_article_content requires a 'content' argument"
        })

    # Check argument types
    if args:
        content = args[0]
        if not isinstance(content, str):
            validation["arg_validation"]["valid"] = False
            validation["arg_validation"]["issues"].append({
                "type": "invalid_argument_type",
                "argument": "content",
                "expected_type": "str",
                "actual_type": type(content).__name__,
                "message": "content argument must be a string"
            })

    if "content" in kwargs:
        content = kwargs["content"]
        if not isinstance(content, str):
            validation["arg_validation"]["valid"] = False
            validation["arg_validation"]["issues"].append({
                "type": "invalid_argument_type",
                "argument": "content",
                "expected_type": "str",
                "actual_type": type(content).__name__,
                "message": "content argument must be a string"
            })

    # Check optional search_query argument
    if len(args) > 1 or "search_query" in kwargs:
        search_query = args[1] if len(args) > 1 else kwargs.get("search_query")
        if search_query is not None and not isinstance(search_query, str):
            validation["arg_validation"]["warnings"].append({
                "type": "invalid_argument_type",
                "argument": "search_query",
                "expected_type": "str",
                "actual_type": type(search_query).__name__,
                "message": "search_query argument should be a string (will be converted if possible)"
            })

    return validation


def _validate_apply_modern_cleaning_args(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Validate arguments for apply_modern_cleaning method."""

    validation = {
        "arg_validation": {
            "valid": True,
            "issues": [],
            "warnings": []
        }
    }

    # Check required arguments
    if len(args) < 1 and "content" not in kwargs:
        validation["arg_validation"]["valid"] = False
        validation["arg_validation"]["issues"].append({
            "type": "missing_required_argument",
            "argument": "content",
            "message": "apply_modern_cleaning requires a 'content' argument"
        })

    # Check argument types
    if args:
        content = args[0]
        if not isinstance(content, str):
            validation["arg_validation"]["valid"] = False
            validation["arg_validation"]["issues"].append({
                "type": "invalid_argument_type",
                "argument": "content",
                "expected_type": "str",
                "actual_type": type(content).__name__,
                "message": "content argument must be a string"
            })

    if "content" in kwargs:
        content = kwargs["content"]
        if not isinstance(content, str):
            validation["arg_validation"]["valid"] = False
            validation["arg_validation"]["issues"].append({
                "type": "invalid_argument_type",
                "argument": "content",
                "expected_type": "str",
                "actual_type": type(content).__name__,
                "message": "content argument must be a string"
            })

    return validation


def validate_content_cleaner_imports(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate that a Python file correctly imports and uses ModernWebContentCleaner.

    Args:
        file_path: Path to the Python file to validate

    Returns:
        Dictionary with validation results
    """

    file_path = Path(file_path)

    if not file_path.exists():
        return {
            "valid": False,
            "error": f"File not found: {file_path}",
            "file_path": str(file_path)
        }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {
            "valid": False,
            "error": f"Failed to read file: {e}",
            "file_path": str(file_path)
        }

    validation_result = {
        "file_path": str(file_path),
        "valid": True,
        "imports": [],
        "method_calls": [],
        "issues": [],
        "suggestions": []
    }

    # Check for ModernWebContentCleaner imports
    import_patterns = [
        r"from\s+.*modern_content_cleaner\s+import\s+.*ModernWebContentCleaner",
        r"import\s+.*modern_content_cleaner",
        r"ModernWebContentCleaner"
    ]

    has_import = False
    for pattern in import_patterns:
        import re
        if re.search(pattern, content):
            has_import = True
            validation_result["imports"].append(pattern)
            break

    if not has_import:
        validation_result["issues"].append({
            "type": "missing_import",
            "message": "ModernWebContentCleaner import not found",
            "suggestion": "Add: from multi_agent_research_system.utils.modern_content_cleaner import ModernWebContentCleaner"
        })

    # Check for method calls that might be incorrect
    method_call_patterns = [
        r"content_cleaner\.(\w+)\s*\(",
        r"ModernWebContentCleaner.*?\.(\w+)\s*\("
    ]

    for pattern in method_call_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            method_name = match.group(1)
            validation_result["method_calls"].append({
                "method": method_name,
                "line": content[:match.start()].count('\n') + 1,
                "context": match.group(0)
            })

            # Validate specific method calls
            if method_name in ["clean_content", "clean", "process_content", "parse_content"]:
                validation_result["issues"].append({
                    "type": "incorrect_method_call",
                    "method": method_name,
                    "line": content[:match.start()].count('\n') + 1,
                    "message": f"Incorrect method '{method_name}' called on ModernWebContentCleaner",
                    "suggestion": "Use 'clean_article_content' instead"
                })

    return validation_result


def fix_content_cleaner_usage(file_path: Union[str, Path],
                            dry_run: bool = True) -> Dict[str, Any]:
    """
    Fix common content cleaner usage issues in a Python file.

    Args:
        file_path: Path to the Python file to fix
        dry_run: If True, show what would be changed without making changes

    Returns:
        Dictionary with fix results
    """

    validation_result = validate_content_cleaner_imports(file_path)

    if not validation_result["issues"]:
        return {
            "file_path": str(file_path),
            "fixed": False,
            "reason": "No issues found to fix",
            "validation": validation_result
        }

    file_path = Path(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        return {
            "file_path": str(file_path),
            "fixed": False,
            "error": f"Failed to read file: {e}",
            "validation": validation_result
        }

    fixed_content = original_content
    changes_made = []

    # Fix incorrect method calls
    method_corrections = {
        "clean_content": "clean_article_content",
        "clean": "clean_article_content",
        "process_content": "clean_article_content",
        "parse_content": "clean_article_content"
    }

    import re
    for incorrect_method, correct_method in method_corrections.items():
        pattern = rf"(\.cleaner\.{incorrect_method}\s*\()"
        if re.search(pattern, fixed_content):
            fixed_content = re.sub(pattern, f"\\1clean_article_content(", fixed_content)
            changes_made.append({
                "type": "method_correction",
                "incorrect": incorrect_method,
                "correct": correct_method
            })

    if changes_made and not dry_run:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
        except Exception as e:
            return {
                "file_path": str(file_path),
                "fixed": False,
                "error": f"Failed to write file: {e}",
                "changes_attempted": changes_made
            }

    return {
        "file_path": str(file_path),
        "fixed": len(changes_made) > 0,
        "dry_run": dry_run,
        "changes_made": changes_made,
        "validation": validation_result
    }


# Hook interface for Claude Agent SDK
def pre_method_call(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook called before method execution to validate content cleaner usage.

    Args:
        context: Context dictionary containing method call information

    Returns:
        Validation result and any corrections needed
    """

    method_name = context.get("method_name")
    instance = context.get("instance")
    args = context.get("args", ())
    kwargs = context.get("kwargs", {})

    if not method_name or not instance:
        return {
            "valid": True,
            "reason": "Insufficient context for validation"
        }

    # Check if this is a ModernWebContentCleaner method call
    if instance.__class__.__name__ == "ModernWebContentCleaner":
        validation_result = validate_modern_web_content_cleaner_usage(
            instance, method_name, args, kwargs
        )

        if not validation_result["valid"]:
            # Apply corrections if available
            if validation_result["corrections"]:
                correction = validation_result["corrections"][0]
                if correction["type"] == "incorrect_method_name":
                    # Return corrected method call
                    return {
                        "valid": False,
                        "needs_correction": True,
                        "correction": correction,
                        "original_method": method_name,
                        "corrected_method": correction["correct_method"],
                        "reason": "Incorrect method name detected and corrected"
                    }

        return {
            "valid": validation_result["valid"],
            "validation_result": validation_result,
            "reason": "Content cleaner method validation"
        }

    return {
        "valid": True,
        "reason": "Not a ModernWebContentCleaner method call"
    }


# Export hook functions
__all__ = [
    "validate_modern_web_content_cleaner_usage",
    "validate_content_cleaner_imports",
    "fix_content_cleaner_usage",
    "pre_method_call"
]