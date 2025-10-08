"""
Research Targets Configuration System

This module provides centralized configuration for research targets across different
scopes and stages, ensuring consistent behavior across the system without hard-coding.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Research targets configuration
RESEARCH_TARGETS = {
    "brief": {
        "primary_clean_scrape_target": 10,
        "editorial_clean_scrape_target": 4,
        "attempt_multiplier": 1.5
    },
    "default": {
        "primary_clean_scrape_target": 15,
        "editorial_clean_scrape_target": 6,
        "attempt_multiplier": 1.5
    },
    "comprehensive": {
        "primary_clean_scrape_target": 20,
        "editorial_clean_scrape_target": 8,
        "attempt_multiplier": 1.5
    }
}

def get_research_targets(scope: str) -> Dict[str, Any]:
    """
    Get research targets configuration for a given scope.

    Args:
        scope: Research scope ("brief", "default", "comprehensive")

    Returns:
        Dictionary containing target configuration

    Raises:
        ValueError: If scope is not supported
    """
    if scope not in RESEARCH_TARGETS:
        logger.warning(f"Unknown scope '{scope}', falling back to 'default'")
        scope = "default"

    return RESEARCH_TARGETS[scope].copy()

def get_scrape_budget(scope: str, stage: str) -> tuple[int, int]:
    """
    Calculate scrape budget for a given scope and stage.

    Args:
        scope: Research scope ("brief", "default", "comprehensive")
        stage: Research stage ("primary", "editorial")

    Returns:
        Tuple of (clean_scrape_target, max_attempts)

    Raises:
        ValueError: If scope or stage is not supported
    """
    targets = get_research_targets(scope)

    if stage == "primary":
        target = targets["primary_clean_scrape_target"]
    elif stage == "editorial":
        target = targets["editorial_clean_scrape_target"]
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 'primary' or 'editorial'")

    max_attempts = int(target * targets["attempt_multiplier"])

    logger.debug(f"Budget for {scope}/{stage}: target={target}, max_attempts={max_attempts}")
    return target, max_attempts

def validate_scope(scope: str) -> str:
    """
    Validate and normalize scope parameter.

    Args:
        scope: Input scope string

    Returns:
        Normalized valid scope string
    """
    valid_scopes = list(RESEARCH_TARGETS.keys())

    if scope not in valid_scopes:
        logger.warning(f"Invalid scope '{scope}', valid scopes: {valid_scopes}, using 'default'")
        return "default"

    return scope

def get_all_scopes() -> list[str]:
    """
    Get list of all available scopes.

    Returns:
        List of valid scope names
    """
    return list(RESEARCH_TARGETS.keys())

def print_research_targets():
    """Print all research targets configuration for debugging."""
    print("Research Targets Configuration:")
    print("=" * 50)
    for scope, config in RESEARCH_TARGETS.items():
        primary_target = config["primary_clean_scrape_target"]
        editorial_target = config["editorial_clean_scrape_target"]
        multiplier = config["attempt_multiplier"]

        primary_attempts = int(primary_target * multiplier)
        editorial_attempts = int(editorial_target * multiplier)

        print(f"\n{scope.upper()}:")
        print(f"  Primary:      {primary_target} successful → {primary_attempts} max attempts")
        print(f"  Editorial:    {editorial_target} successful → {editorial_attempts} max attempts")
        print(f"  Multiplier:   {multiplier}x")

if __name__ == "__main__":
    # Test the configuration system
    print_research_targets()

    # Test budget calculations
    print("\nBudget Calculation Tests:")
    print("=" * 30)
    for scope in get_all_scopes():
        for stage in ["primary", "editorial"]:
            target, attempts = get_scrape_budget(scope, stage)
            print(f"{scope}/{stage}: {target} target → {attempts} attempts")