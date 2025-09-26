import ast
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from git import Repo
from tqdm import tqdm


def process_file_change(repo_path, commit_hexsha, parent_hexsha, file_path):
    """Process a single file change to extract AST content."""
    try:
        repo = Repo(repo_path)
        old_file_content = ast.dump(
            ast.parse(repo.git.show(f"{parent_hexsha}:{file_path}")),
            include_attributes=True,
        )
        current_file_content = ast.dump(
            ast.parse(repo.git.show(f"{commit_hexsha}:{file_path}")),
            include_attributes=True,
        )
        return {
            "Path": file_path,
            "oldFileContent": old_file_content,
            "currentFileContent": current_file_content,
        }
    except Exception:
        return None


# get the the changes in the latest commits and store them in a dataframe
def last_commit_changes(repo_path):
    repo = Repo(repo_path)
    modified_items = list(repo.head.commit.diff("HEAD~1").iter_change_type("M"))
    python_files = [
        item.a_path for item in modified_items if item.a_path.endswith(".py")
    ]

    if not python_files:
        return pd.DataFrame()

    # Use parallel processing for file changes
    max_workers = min(multiprocessing.cpu_count(), len(python_files))
    modified_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_file_change,
                repo_path,
                repo.head.commit.hexsha,
                repo.head.commit.parents[0].hexsha
                if repo.head.commit.parents
                else None,
                file_path,
            ): file_path
            for file_path in python_files
        }

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                modified_files.append(result)

    df = pd.DataFrame(modified_files)
    output_file = Path(repo_path) / "changes.csv"
    df.to_csv(output_file, index=False)
    return df


def all_commits(repo_path, specific_commits=None, **iter_commits_kwargs):
    """
    Extract commit history and changes from a Git repository.

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
    repo = Repo(repo_path)
    path_to_create = Path(repo_path) / "changes"
    path_to_create.mkdir(exist_ok=True)

    # Filter commits more efficiently
    if specific_commits is not None:
        specific_commits_set = set(specific_commits)
        commits = [
            commit
            for commit in repo.iter_commits(**iter_commits_kwargs)
            if str(commit) in specific_commits_set
        ]
    else:
        commits = list(repo.iter_commits(**iter_commits_kwargs))

    # Filter out merge commits and initial commits upfront
    valid_commits = [commit for commit in commits if len(commit.parents) == 1]

    print(f"Processing {len(valid_commits)} valid commits...")

    # Process commits in batches for better memory usage
    batch_size = min(10, multiprocessing.cpu_count())
    max_workers = min(
        4, multiprocessing.cpu_count()
    )  # Limit workers to avoid Git conflicts

    for i in tqdm(
        range(0, len(valid_commits), batch_size),
        desc="Processing commit batches",
        unit="batch",
    ):
        batch_commits = valid_commits[i : i + batch_size]

        for commit in batch_commits:
            # Get modified Python files for this commit
            modified_items = list(commit.diff(commit.parents[0]).iter_change_type("M"))
            python_files = [
                item.a_path for item in modified_items if item.a_path.endswith(".py")
            ]

            if not python_files:
                continue

            # Process files in parallel
            modified_files = []
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


def repo_changes_args(args):
    if args.lastcommit:
        last_commit_changes(args.path)
    if args.allcommits:
        # Extract iter_commits options from args if available
        iter_commits_kwargs = {}
        if hasattr(args, "max_count") and args.max_count:
            iter_commits_kwargs["max_count"] = args.max_count
        if hasattr(args, "skip") and args.skip:
            iter_commits_kwargs["skip"] = args.skip
        if hasattr(args, "since") and args.since:
            iter_commits_kwargs["since"] = args.since
        if hasattr(args, "until") and args.until:
            iter_commits_kwargs["until"] = args.until
        if hasattr(args, "rev") and args.rev:
            iter_commits_kwargs["rev"] = args.rev
        if hasattr(args, "paths") and args.paths:
            iter_commits_kwargs["paths"] = args.paths

        all_commits(args.path, **iter_commits_kwargs)
