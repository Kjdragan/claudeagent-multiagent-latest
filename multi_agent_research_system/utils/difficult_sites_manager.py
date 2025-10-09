#!/usr/bin/env python3
"""
Standalone Difficult Sites Manager

A lightweight manager for difficult website domains that doesn't require full dependencies.
Used for CLI operations and basic management of difficult sites configuration.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DifficultSite:
    """Configuration for a difficult website domain."""

    domain: str
    level: int
    reason: str
    last_updated: str

    def __post_init__(self):
        """Validate the difficult site configuration."""
        if not 0 <= self.level <= 3:
            raise ValueError(f"Anti-bot level must be 0-3, got {self.level}")
        if not self.domain:
            raise ValueError("Domain cannot be empty")


class StandaloneDifficultSitesManager:
    """Standalone manager for difficult website domains."""

    def __init__(self, config_file: str | None = None):
        """Initialize the difficult sites manager.

        Args:
            config_file: Path to the difficult sites JSON config file.
                        If None, uses default location in utils directory.
        """
        if config_file is None:
            # Default location in the utils directory
            self.config_file = Path(__file__).parent / "difficult_sites.json"
        else:
            self.config_file = Path(config_file)

        self._difficult_sites: dict[str, DifficultSite] = {}
        self._config_metadata: dict[str, Any] = {}
        self._load_difficult_sites()

    def _load_difficult_sites(self):
        """Load difficult sites from the JSON configuration file."""
        try:
            if not self.config_file.exists():
                logger.warning(f"Difficult sites config file not found: {self.config_file}")
                return

            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load difficult sites
            if "difficult_sites" in data:
                for domain, site_data in data["difficult_sites"].items():
                    try:
                        site = DifficultSite(
                            domain=domain,
                            level=site_data["level"],
                            reason=site_data["reason"],
                            last_updated=site_data["last_updated"]
                        )
                        self._difficult_sites[domain.lower()] = site
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Invalid difficult site entry for {domain}: {e}")

            # Load metadata
            self._config_metadata = data.get("metadata", {})

            logger.info(f"Loaded {len(self._difficult_sites)} difficult sites from {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to load difficult sites config: {e}")
            self._difficult_sites = {}
            self._config_metadata = {}

    def save_difficult_sites(self):
        """Save difficult sites to the JSON configuration file."""
        try:
            # Prepare data structure
            data = {
                "difficult_sites": {},
                "metadata": {
                    **self._config_metadata,
                    "last_saved": datetime.now().isoformat()
                }
            }

            # Convert DifficultSite objects to dictionaries
            for domain, site in self._difficult_sites.items():
                data["difficult_sites"][domain] = {
                    "level": site.level,
                    "reason": site.reason,
                    "last_updated": site.last_updated
                }

            # Update usage stats
            level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for site in self._difficult_sites.values():
                level_counts[site.level] += 1

            data["metadata"]["usage_stats"] = {
                "total_sites": len(self._difficult_sites),
                "level_distribution": level_counts
            }

            # Write to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self._difficult_sites)} difficult sites to {self.config_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save difficult sites config: {e}")
            return False

    def get_difficult_site(self, domain: str) -> DifficultSite | None:
        """Get difficult site configuration for a domain.

        Args:
            domain: The domain to look up (case-insensitive)

        Returns:
            DifficultSite configuration if found, None otherwise
        """
        return self._difficult_sites.get(domain.lower())

    def is_difficult_site(self, domain: str) -> bool:
        """Check if a domain is configured as a difficult site.

        Args:
            domain: The domain to check (case-insensitive)

        Returns:
            True if domain is a difficult site, False otherwise
        """
        return domain.lower() in self._difficult_sites

    def add_difficult_site(self, domain: str, level: int, reason: str) -> bool:
        """Add a new difficult site to the configuration.

        Args:
            domain: The domain to add
            level: Anti-bot level (0-3)
            reason: Reason for the difficulty level

        Returns:
            True if added successfully, False otherwise
        """
        try:
            site = DifficultSite(
                domain=domain.lower(),
                level=level,
                reason=reason,
                last_updated=datetime.now().isoformat()
            )

            self._difficult_sites[domain.lower()] = site
            logger.info(f"Added difficult site: {domain.lower()} (level {level}: {reason})")
            return self.save_difficult_sites()

        except ValueError as e:
            logger.error(f"Failed to add difficult site {domain}: {e}")
            return False

    def remove_difficult_site(self, domain: str) -> bool:
        """Remove a difficult site from the configuration.

        Args:
            domain: The domain to remove

        Returns:
            True if removed successfully, False otherwise
        """
        domain_lower = domain.lower()
        if domain_lower in self._difficult_sites:
            del self._difficult_sites[domain_lower]
            logger.info(f"Removed difficult site: {domain_lower}")
            return self.save_difficult_sites()
        else:
            logger.warning(f"Difficult site not found for removal: {domain}")
            return False

    def get_all_difficult_sites(self) -> dict[str, DifficultSite]:
        """Get all difficult site configurations.

        Returns:
            Dictionary mapping domains to DifficultSite objects
        """
        return self._difficult_sites.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about difficult sites configuration.

        Returns:
            Dictionary with configuration statistics
        """
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for site in self._difficult_sites.values():
            level_counts[site.level] += 1

        return {
            "total_sites": len(self._difficult_sites),
            "level_distribution": level_counts,
            "config_file": str(self.config_file),
            "last_loaded": datetime.now().isoformat(),
            "metadata": self._config_metadata
        }


def extract_domain_from_url(url: str) -> str:
    """Extract domain from URL for difficult sites matching.

    Args:
        url: The URL to extract domain from

    Returns:
        Domain string in lowercase, or empty string if extraction fails
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception as e:
        logger.warning(f"Failed to extract domain from URL '{url}': {e}")
        return ""


def is_url_difficult_site(url: str, manager: StandaloneDifficultSitesManager | None = None) -> tuple[bool, DifficultSite | None]:
    """Check if a URL belongs to a difficult site and get the site configuration.

    Args:
        url: The URL to check
        manager: Optional manager to use (creates one if None)

    Returns:
        Tuple of (is_difficult, site_config)
    """
    if manager is None:
        manager = StandaloneDifficultSitesManager()

    domain = extract_domain_from_url(url)
    if not domain:
        return False, None

    difficult_site = manager.get_difficult_site(domain)
    return (difficult_site is not None), difficult_site


def get_predefined_anti_bot_level(url: str, manager: StandaloneDifficultSitesManager | None = None) -> int | None:
    """Get the predefined anti-bot level for a URL if it's a difficult site.

    Args:
        url: The URL to check
        manager: Optional manager to use (creates one if None)

    Returns:
        Anti-bot level (0-3) if URL is a difficult site, None otherwise
    """
    is_difficult, difficult_site = is_url_difficult_site(url, manager)
    return difficult_site.level if is_difficult and difficult_site else None


# Create a global instance for easy access
_global_manager: StandaloneDifficultSitesManager | None = None


def get_standalone_manager() -> StandaloneDifficultSitesManager:
    """Get or create global standalone manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = StandaloneDifficultSitesManager()
    return _global_manager