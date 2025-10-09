#!/usr/bin/env python3
"""
Simple Difficult Sites CLI Tool

Command-line interface for managing difficult website domains and their anti-bot levels.
This version uses the standalone manager to avoid dependency issues.

Usage:
    python difficult_sites_cli_simple.py list
    python difficult_sites_cli_simple.py add linkedin.com 2 "Professional network requires advanced simulation"
    python difficult_sites_cli_simple.py remove linkedin.com
    python difficult_sites_cli_simple.py stats
    python difficult_sites_cli_simple.py check https://linkedin.com/in/some-profile
"""

import argparse
import sys
from typing import Any

from difficult_sites_manager import (
    get_standalone_manager,
    extract_domain_from_url,
    is_url_difficult_site,
    DifficultSite
)


def format_difficult_site_table(sites: dict[str, DifficultSite]) -> str:
    """Format difficult sites into a readable table.

    Args:
        sites: Dictionary of difficult sites from the manager

    Returns:
        Formatted table string
    """
    if not sites:
        return "No difficult sites configured."

    # Calculate column widths
    max_domain_len = max(len(domain) for domain in sites.keys())
    max_reason_len = max(len(site.reason) for site in sites.values())

    domain_width = max(max_domain_len, 25)
    level_width = 6
    reason_width = max(max_reason_len, 50)

    # Create header
    header = f"{'DOMAIN':<{domain_width}} {'LEVEL':<{level_width}} {'REASON':<{reason_width}} {'LAST_UPDATED'}"
    separator = f"{'-' * domain_width} {'-' * level_width} {'-' * reason_width} {'-' * 19}"

    # Create rows
    rows = []
    for domain, site in sites.items():
        rows.append(f"{domain:<{domain_width}} {site.level:<{level_width}} {site.reason:<{reason_width}} {site.last_updated}")

    return f"{header}\n{separator}\n" + "\n".join(rows)


def cmd_list(args):
    """List all difficult sites."""
    manager = get_standalone_manager()
    sites = manager.get_all_difficult_sites()

    if args.verbose:
        # Verbose mode with full details
        print("🎯 DIFFICULT SITES CONFIGURATION")
        print("=" * 80)
        print(format_difficult_site_table(sites))
        print(f"\nTotal sites: {len(sites)}")

        # Show level distribution
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for site in sites.values():
            level_counts[site.level] += 1

        print("\nLevel Distribution:")
        level_names = ["Basic (0)", "Enhanced (1)", "Advanced (2)", "Stealth (3)"]
        for i, (count, name) in enumerate(zip(level_counts.values(), level_names)):
            if count > 0:
                print(f"  {name}: {count} sites")
    else:
        # Simple mode - just list domains and levels
        if not sites:
            print("No difficult sites configured.")
        else:
            for domain, site in sites.items():
                print(f"{domain} (level {site.level})")


def cmd_add(args):
    """Add a new difficult site."""
    manager = get_standalone_manager()

    # Validate level
    if not 0 <= args.level <= 3:
        print(f"❌ Error: Level must be 0-3, got {args.level}")
        print("  0 = Basic (Basic SERP API and simple crawl)")
        print("  1 = Enhanced (Enhanced headers + JavaScript rendering)")
        print("  2 = Advanced (Advanced proxy rotation + browser automation)")
        print("  3 = Stealth (Stealth mode with full browser simulation)")
        return 1

    # Add the site
    success = manager.add_difficult_site(args.domain.lower(), args.level, args.reason)

    if success:
        print(f"✅ Added difficult site: {args.domain.lower()} (level {args.level})")
        print(f"   Reason: {args.reason}")

        # Show updated configuration
        sites = manager.get_all_difficult_sites()
        print(f"\n📊 Total difficult sites configured: {len(sites)}")
    else:
        print(f"❌ Failed to add difficult site: {args.domain}")
        return 1


def cmd_remove(args):
    """Remove a difficult site."""
    manager = get_standalone_manager()

    # Check if site exists
    if not manager.is_difficult_site(args.domain.lower()):
        print(f"⚠️  Site '{args.domain.lower()}' is not in the difficult sites list")
        return 1

    # Remove the site
    success = manager.remove_difficult_site(args.domain.lower())

    if success:
        print(f"✅ Removed difficult site: {args.domain.lower()}")

        # Show updated configuration
        sites = manager.get_all_difficult_sites()
        print(f"\n📊 Total difficult sites configured: {len(sites)}")
    else:
        print(f"❌ Failed to remove difficult site: {args.domain}")
        return 1


def cmd_stats(args):
    """Show comprehensive statistics."""
    manager = get_standalone_manager()
    stats = manager.get_stats()

    print("📊 DIFFICULT SITES STATISTICS")
    print("=" * 60)

    # Basic stats
    print(f"Config file: {stats['config_file']}")
    print(f"Total difficult sites: {stats['total_sites']}")

    # Level distribution
    print(f"\n📈 LEVEL DISTRIBUTION")
    level_names = ["Basic (0)", "Enhanced (1)", "Advanced (2)", "Stealth (3)"]
    for i, (count, name) in enumerate(zip(stats['level_distribution'].values(), level_names)):
        if count > 0:
            print(f"  {name}: {count} sites")

    # Metadata
    metadata = stats.get('metadata', {})
    if metadata:
        print(f"\n📝 METADATA")
        if 'version' in metadata:
            print(f"  Version: {metadata['version']}")
        if 'description' in metadata:
            print(f"  Description: {metadata['description']}")
        if 'created' in metadata:
            print(f"  Created: {metadata['created']}")


def cmd_check(args):
    """Check if a URL/domain is a difficult site."""
    if not args.url.startswith(('http://', 'https://')):
        args.url = f'https://{args.url}'

    domain = extract_domain_from_url(args.url)
    if not domain:
        print(f"❌ Invalid URL: {args.url}")
        return 1

    manager = get_standalone_manager()
    is_difficult, site_config = is_url_difficult_site(args.url, manager)

    print(f"🔍 CHECKING: {args.url}")
    print(f"📡 Domain: {domain}")

    if is_difficult and site_config:
        print(f"✅ DIFFICULT SITE DETECTED")
        print(f"   Level: {site_config.level}")
        print(f"   Reason: {site_config.reason}")
        print(f"   Last updated: {site_config.last_updated}")
    else:
        print(f"❌ Not in difficult sites list")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage difficult website domains for anti-bot optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                    # List all difficult sites
  %(prog)s add linkedin.com 2 "Professional network requires advanced simulation"
  %(prog)s remove linkedin.com     # Remove a difficult site
  %(prog)s check https://linkedin.com/in/profile  # Check if URL is difficult site
  %(prog)s stats                   # Show statistics

Anti-bot Levels:
  0 = Basic (Basic SERP API and simple crawl)
  1 = Enhanced (Enhanced headers + JavaScript rendering)
  2 = Advanced (Advanced proxy rotation + browser automation)
  3 = Stealth (Stealth mode with full browser simulation)
        """
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # List command
    subparsers.add_parser('list', help='List all difficult sites')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new difficult site')
    add_parser.add_argument('domain', help='Domain name (e.g., linkedin.com)')
    add_parser.add_argument('level', type=int, help='Anti-bot level (0-3)')
    add_parser.add_argument('reason', help='Reason for the difficulty level')

    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a difficult site')
    remove_parser.add_argument('domain', help='Domain name to remove')

    # Stats command
    subparsers.add_parser('stats', help='Show statistics')

    # Check command
    check_parser = subparsers.add_parser('check', help='Check if a URL/domain is a difficult site')
    check_parser.add_argument('url', help='URL or domain to check')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        if args.command == 'list':
            cmd_list(args)
        elif args.command == 'add':
            return cmd_add(args)
        elif args.command == 'remove':
            return cmd_remove(args)
        elif args.command == 'stats':
            cmd_stats(args)
        elif args.command == 'check':
            return cmd_check(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())