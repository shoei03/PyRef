import ast
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from git import Repo
from tqdm import tqdm

# Cache for repository objects to avoid repeated instantiation
_repo_cache: Dict[str, Repo] = {}


def get_repo(repo_path: str) -> Repo:
    """Get or create a cached repository object."""
    if repo_path not in _repo_cache:
        _repo_cache[repo_path] = Repo(repo_path)
    return _repo_cache[repo_path]


def clear_repo_cache() -> None:
    """Clear the repository cache to free memory."""
    global _repo_cache
    _repo_cache.clear()
    parse_file_content.cache_clear()


@lru_cache(maxsize=256)
def parse_file_content(content: str) -> Optional[str]:
    """Parse Python file content to AST with caching."""
    try:
        return ast.dump(ast.parse(content), include_attributes=True)
    except (SyntaxError, ValueError):
        return None


def get_file_content(repo: Repo, commit_hexsha: str, file_path: str) -> Optional[str]:
    """Get file content from a specific commit with error handling."""
    try:
        # Use git cat-file for fastest access - faster than git show
        return repo.git.execute(
            ["git", "cat-file", "-p", f"{commit_hexsha}:{file_path}"]
        )
    except Exception:
        return None


def process_file_change(
    repo_path: str, commit_hexsha: str, parent_hexsha: str, file_path: str
) -> Optional[Dict[str, str]]:
    """Process a single file change to extract AST content with improved error handling."""
    if not parent_hexsha:  # Skip if no parent commit
        return None

    try:
        repo = get_repo(repo_path)

        # Get file contents
        old_content = get_file_content(repo, parent_hexsha, file_path)
        current_content = get_file_content(repo, commit_hexsha, file_path)

        if not old_content or not current_content:
            return None

        # Parse contents to AST
        old_ast = parse_file_content(old_content)
        current_ast = parse_file_content(current_content)

        if not old_ast or not current_ast:
            return None

        return {
            "Path": file_path,
            "oldFileContent": old_ast,
            "currentFileContent": current_ast,
        }
    except Exception:
        return None


def get_python_files_from_diff(diff_items) -> List[str]:
    """Extract Python file paths from diff items."""
    return [
        item.a_path
        for item in diff_items
        if item.a_path and item.a_path.endswith(".py")
    ]


def last_commit_changes(repo_path: str) -> pd.DataFrame:
    """Get the changes in the latest commit and store them in a dataframe."""
    repo = get_repo(repo_path)

    # Check if HEAD has parent commits
    if not repo.head.commit.parents:
        return pd.DataFrame()

    modified_items = list(repo.head.commit.diff("HEAD~1").iter_change_type("M"))
    python_files = get_python_files_from_diff(modified_items)

    if not python_files:
        return pd.DataFrame()

    # Optimize worker count based on file count and CPU cores
    max_workers = min(multiprocessing.cpu_count(), len(python_files), 8)  # Cap at 8
    modified_files: List[Dict[str, str]] = []

    parent_hexsha = (
        repo.head.commit.parents[0].hexsha if repo.head.commit.parents else None
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_file_change,
                repo_path,
                repo.head.commit.hexsha,
                parent_hexsha,
                file_path,
            ): file_path
            for file_path in python_files
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files"
        ):
            result = future.result()
            if result is not None:
                modified_files.append(result)

    if not modified_files:
        return pd.DataFrame()

    df = pd.DataFrame(modified_files)
    output_file = Path(repo_path) / "changes.csv"
    df.to_csv(output_file, index=False)
    return df


def filter_valid_commits(
    commits, specific_commits: Optional[Set[str]] = None
) -> List[Any]:
    """Filter commits to only include valid non-merge commits."""
    valid_commits = []
    specific_set = set(specific_commits) if specific_commits else None

    for commit in commits:
        # Skip merge commits and commits without parents
        if len(commit.parents) != 1:
            continue

        # If specific commits are provided, only include those
        if specific_set and str(commit) not in specific_set:
            continue

        valid_commits.append(commit)

    return valid_commits


def process_commit_batch(
    repo_path: str, commits: List[Any], path_to_create: Path, max_workers: int
) -> None:
    """Process a batch of commits in parallel."""
    for commit in commits:
        # Get modified Python files for this commit
        modified_items = list(commit.diff(commit.parents[0]).iter_change_type("M"))
        python_files = get_python_files_from_diff(modified_items)

        if not python_files:
            continue

        # Process files in parallel with limited workers
        modified_files: List[Dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_file_change,
                    repo_path,
                    commit.hexsha,
                    commit.parents[0].hexsha,
                    file_path,
                ): file_path
                for file_path in python_files
            }

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    modified_files.append(result)

        # Save results if any files were processed successfully
        if modified_files:
            df = pd.DataFrame(modified_files)
            changes_file = path_to_create / f"{commit}.csv"
            df.to_csv(changes_file, index=False)


def all_commits(
    repo_path: str,
    specific_commits: Optional[List[str]] = None,
    **iter_commits_kwargs: Any,
) -> None:
    """
    Extract commit history and changes from a Git repository with improved performance.

    Args:
        repo_path: Path to the git repository
        specific_commits: List of specific commit hashes to process (optional)
        **iter_commits_kwargs: Additional keyword arguments for repo.iter_commits()
                               Common options:
                               - max_count: Maximum number of commits to retrieve
                               - skip: Number of commits to skip
                               - since: Only commits after this date
                               - until: Only commits before this date
                               - paths: Only commits that modified these paths
                               - rev: Revision or branch to start from (default: HEAD)
    """
    repo = get_repo(repo_path)
    path_to_create = Path(repo_path) / "changes"
    path_to_create.mkdir(exist_ok=True)

    # Convert specific_commits to set for O(1) lookup if provided
    specific_commits_set = set(specific_commits) if specific_commits else None

    # Get all commits first (lazy evaluation)
    all_commits_iter = repo.iter_commits(**iter_commits_kwargs)

    # Filter valid commits efficiently
    valid_commits = filter_valid_commits(all_commits_iter, specific_commits_set)

    if not valid_commits:
        print("No valid commits found to process.")
        return

    print(f"Processing {len(valid_commits)} valid commits...")

    # Optimize batch size and worker count
    cpu_count = multiprocessing.cpu_count()
    batch_size = min(max(5, cpu_count // 2), 15)  # Adaptive batch size
    max_workers = min(3, cpu_count)  # Conservative to avoid Git conflicts

    # Process commits in batches for better memory usage
    for i in tqdm(
        range(0, len(valid_commits), batch_size),
        desc="Processing commit batches",
        unit="batch",
    ):
        batch_commits = valid_commits[i : i + batch_size]
        process_commit_batch(repo_path, batch_commits, path_to_create, max_workers)


def repo_changes_args(args: Any) -> None:
    """Process repository changes based on command line arguments."""
    if hasattr(args, "lastcommit") and args.lastcommit:
        last_commit_changes(args.path)

    if hasattr(args, "allcommits") and args.allcommits:
        # Extract iter_commits options from args efficiently
        iter_commits_kwargs = {}

        # Map of argument names to their corresponding iter_commits parameters
        arg_mapping = {
            "max_count": "max_count",
            "skip": "skip",
            "since": "since",
            "until": "until",
            "rev": "rev",
            "paths": "paths",
        }

        for arg_name, kwarg_name in arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    iter_commits_kwargs[kwarg_name] = value

        all_commits(args.path, **iter_commits_kwargs)
