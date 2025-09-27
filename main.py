"""
PyRef - Python Refactoring Analysis Tool

A command-line tool for analyzing refactoring patterns in Python repositories.
Provides functionality for repository management, diff analysis, and refactoring extraction.
"""

import argparse
import sys

# =============================================================================
# Command Handler Functions
# =============================================================================


def clone_repo_access(args):
    """Handle repository cloning command."""
    from repomanager import repo_utils

    repo_utils.clone_repo_args(args)


def repo_changes_access(args):
    """Handle repository changes analysis command."""
    from repomanager import repo_changes

    repo_changes.repo_changes_args(args)


def build_diff_lists_access(args):
    """Handle diff list building command."""
    from preprocessing import diff_list

    diff_list.build_diff_lists_args(args)


def validate_results(args):
    """Handle results validation command."""
    from preprocessing import diff_list

    diff_list.validate(args)


def extract_refs(args):
    """Handle refactoring extraction command."""
    from preprocessing import diff_list

    diff_list.extract_refs(args)


# =============================================================================
# Argument Parser Setup
# =============================================================================


def setup_argument_parser():
    """Set up and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="PyRef - Python Refactoring Analysis Tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Repository cloning command
    clone_repo = subparsers.add_parser("repoClone", help="Clone a repository")
    clone_repo.add_argument("-u", "--username", required=True, help="Repository owner")
    clone_repo.add_argument("-r", "--reponame", required=True, help="Repository name")
    clone_repo.set_defaults(func=clone_repo_access)

    # Repository changes analysis command
    repo_changes = subparsers.add_parser(
        "repoChanges", help="Analyze changes in repository"
    )
    repo_changes.add_argument(
        "-p", "--path", required=True, help="Path to the repository"
    )
    repo_changes.add_argument(
        "-l", "--lastcommit", action="store_true", help="Changes between last commits"
    )
    repo_changes.add_argument(
        "-al", "--allcommits", action="store_true", help="Changes among all commits"
    )
    repo_changes.add_argument(
        "--max-count", type=int, help="Maximum number of commits to retrieve"
    )
    repo_changes.add_argument(
        "--skip", type=int, help="Number of commits to skip from the beginning"
    )
    repo_changes.add_argument(
        "--since",
        help="Only commits after this date (e.g., '2023-01-01' or '1 week ago')",
    )
    repo_changes.add_argument(
        "--until",
        help="Only commits before this date (e.g., '2023-12-31' or '1 day ago')",
    )
    repo_changes.add_argument(
        "--rev", help="Revision or branch to start from (default: HEAD)"
    )
    repo_changes.add_argument(
        "--paths", nargs="+", help="Only commits that modified these file paths"
    )
    repo_changes.set_defaults(func=repo_changes_access)

    # Diff list building command
    diff_list = subparsers.add_parser("reflist", help="Build the diff lists")
    diff_list.add_argument("-p", "--path", required=True, help="Path to the CSV file")
    diff_list.add_argument(
        "-c", "--commit", required=False, help="Specific commit hash"
    )
    diff_list.set_defaults(func=build_diff_lists_access)

    # Results validation command
    validate_res = subparsers.add_parser("validate", help="Validate analysis results")
    validate_res.add_argument(
        "-p", "--path", required=True, help="Path to the validation CSV file"
    )
    validate_res.set_defaults(func=validate_results)

    # Refactoring extraction command
    extract_ref = subparsers.add_parser("getrefs", help="Extract refactoring patterns")
    extract_ref.add_argument(
        "-r", "--repopath", required=True, help="Path to the repository"
    )
    extract_ref.add_argument(
        "-c", "--commit", required=False, help="Specific commit hash"
    )
    extract_ref.add_argument(
        "-d", "--directory", required=False, help="Specific directories", nargs="+"
    )
    extract_ref.add_argument(
        "-s", "--skip", required=False, help="Skip commit after n minutes"
    )
    extract_ref.add_argument(
        "--max-count", type=int, help="Maximum number of commits to retrieve"
    )
    extract_ref.add_argument(
        "--skip-commits", type=int, help="Number of commits to skip from the beginning"
    )
    extract_ref.add_argument(
        "--since",
        help="Only commits after this date (e.g., '2023-01-01' or '1 week ago')",
    )
    extract_ref.add_argument(
        "--until",
        help="Only commits before this date (e.g., '2023-12-31' or '1 day ago')",
    )
    extract_ref.add_argument(
        "--rev", help="Revision or branch to start from (default: HEAD)"
    )
    extract_ref.add_argument(
        "--paths", nargs="+", help="Only commits that modified these file paths"
    )
    extract_ref.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing even if errors occur",
    )
    extract_ref.set_defaults(func=extract_refs)

    return parser


# =============================================================================
# Main Function
# =============================================================================


def main():
    """Main entry point of the application."""
    # Show help if no arguments provided
    if len(sys.argv) <= 1:
        sys.argv.append("--help")

    # Set up argument parser and parse arguments
    parser = setup_argument_parser()
    options = parser.parse_args()

    try:
        options.func(options)
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user (Ctrl+C)")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        if hasattr(options, "continue_on_error") and options.continue_on_error:
            print("Continuing due to --continue-on-error flag...")
        else:
            print(
                "Use --continue-on-error flag to ignore errors and continue processing"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
