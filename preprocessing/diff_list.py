import hashlib
import json
import signal
import threading
import time
from ast import *
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from preprocessing.conditions_match import *
from preprocessing.revision import Rev
from preprocessing.utils import to_tree


class RepeatedTimer(object):
    # from https://stackoverflow.com/a/40965385
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.next_call = time.time()
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self.next_call += self.interval
            self._timer = threading.Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


def timeout_handler(num, stack):
    print("Commit skipped due to the long processing time")
    raise TimeoutError


def execution_reminder():
    print("Please wait, the process is still running. ", time.ctime())


def build_diff_lists(
    changes_path,
    commit=None,
    directory=None,
    skip_time=None,
    continue_on_error=False,
):
    refactorings = []
    t0 = time.time()
    if commit is not None:
        print(commit)
        name = commit + ".csv"

        try:
            df = pd.read_csv(Path(changes_path) / name)
            if directory is not None:
                df = df[df["Path"].isin(directory)]
            rev_a = Rev()
            rev_b = Rev()
            df.apply(lambda row: populate(row, rev_a, rev_b), axis=1)
            # try:
            rev_difference = rev_a.revision_difference(rev_b)
            refs = rev_difference.get_refactorings()
            for ref in refs:
                refactorings.append((ref, name.split(".")[0]))
        except Exception as e:
            print(f"Error processing commit {commit}: {e}")
            if not continue_on_error:
                raise
            print("Continuing despite error due to continue-on-error flag")
    else:
        changes_dir = Path(changes_path)
        csv_files = [f.name for f in changes_dir.iterdir() if f.suffix == ".csv"]

        # Sort files by size to show progress better
        csv_files_with_size = []
        for name in csv_files:
            file_path = changes_dir / name
            size = file_path.stat().st_size
            csv_files_with_size.append((name, size))
        csv_files_with_size.sort(key=lambda x: x[1])  # Sort by size

        # Choose parallel or sequential processing
        use_parallel = (
            len(csv_files_with_size) > 2
        )  # Use parallel for more than 2 files

        if use_parallel:
            print(
                f"Using parallel processing for {len(csv_files_with_size)} commits..."
            )
            commit_refactorings = process_commits_parallel(
                csv_files_with_size,
                changes_path,
                directory,
                skip_time,
                max_workers=4,
                continue_on_error=continue_on_error,
            )
            refactorings.extend(commit_refactorings)
        else:
            # Original sequential processing for small number of files
            pbar = tqdm(csv_files_with_size, desc="Extracting Refs", unit="file")
            for ind, (name, file_size) in enumerate(pbar):
                start_time_commit = time.time()
                commit_hash = name.split(".")[0][:8]
                file_size_mb = file_size / (1024 * 1024)
                pbar.set_description(f"Processing {commit_hash} ({file_size_mb:.1f}MB)")

                try:
                    df = pd.read_csv(Path(changes_path) / name)
                    if directory is not None:
                        df = df[df["Path"].isin(directory)]

                    pbar.set_postfix(files=len(df), refresh=True)

                    rev_a = Rev()
                    rev_b = Rev()
                    df.apply(lambda row: populate(row, rev_a, rev_b), axis=1)
                    if skip_time is not None:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(float(skip_time) * 60))
                    rt = RepeatedTimer(480, execution_reminder)
                    try:
                        rev_difference = rev_a.revision_difference(rev_b)
                        refs = rev_difference.get_refactorings()
                        for ref in refs:
                            refactorings.append((ref, name.split(".")[0]))
                    except Exception as e:
                        print(f"Failed to process commit {commit_hash}: {e}")
                        if not continue_on_error:
                            raise
                    except TimeoutError:
                        print(
                            f"Commit {commit_hash} skipped due to the long processing time"
                        )
                    finally:
                        rt.stop()
                        if skip_time is not None:
                            signal.alarm(0)

                    elapsed = time.time() - start_time_commit
                    pbar.set_postfix(
                        files=len(df), time=f"{elapsed:.1f}s", refresh=True
                    )

                except Exception as e:
                    print(f"Error processing commit {commit_hash}: {e}")
                    if continue_on_error:
                        print(f"Skipping commit {commit_hash} and continuing...")
                        continue
                    else:
                        raise

    t1 = time.time()
    total = t1 - t0
    print(
        "-----------------------------------------------------------------------------------------------------------"
    )
    print("Total Time:", total)
    print("Total Number of Refactorings:", len(refactorings))
    refactorings.sort(key=lambda x: x[1])
    json_outputs = []
    for ref in refactorings:
        print(f"commit: {ref[1]:>3s} - {str(ref[0]).strip()}")
        data = ref[0].to_json_format()
        data["Commit"] = ref[1]
        json_outputs.append(data)
        # ref[0].to_graph()
    changes_path = Path(changes_path).resolve()
    repo_name = changes_path.parts[
        -2
    ]  # Get the repository name (DummyRef, pandas, etc.)

    # Create output filename
    output_filename = f"{repo_name}_data.json"
    with open(output_filename, "w") as outfile:
        outfile.write(json.dumps(json_outputs, indent=4))

    print(f"Results saved to: {output_filename}")

    return refactorings


def extract_refs(args):
    # owner_name = args.repo.split("/")[0]
    # repo_name = args.repo.split("/")[1]

    from repomanager import repo_changes

    repo_path = args.repopath
    continue_on_error = getattr(args, "continue_on_error", False)

    if args.skip is not None:
        skip_time = args.skip
        print(
            "\nCommit will be skipped if the processing time is longer than",
            skip_time,
            "minutes.",
        )
    else:
        skip_time = None

    if continue_on_error:
        print(
            "\nContinue-on-error mode enabled: will attempt to process all commits despite errors"
        )

    try:
        if args.commit is not None:
            try:
                repo_changes.all_commits(repo_path, [args.commit])
                print("\nExtracting Refs...")
                build_diff_lists(
                    str(Path(repo_path) / "changes"),
                    args.commit,
                    args.directory,
                    skip_time,
                    continue_on_error,
                )
            except Exception as e:
                print(f"Error processing specific commit {args.commit}: {e}")
                if not continue_on_error:
                    raise
        else:
            try:
                print("\nExtracting commit history...")
                # Prepare iter_commits options
                iter_commits_kwargs = {}
                if hasattr(args, "max_count") and args.max_count:
                    iter_commits_kwargs["max_count"] = args.max_count
                if hasattr(args, "skip_commits") and args.skip_commits:
                    iter_commits_kwargs["skip"] = args.skip_commits
                if hasattr(args, "since") and args.since:
                    iter_commits_kwargs["since"] = args.since
                if hasattr(args, "until") and args.until:
                    iter_commits_kwargs["until"] = args.until
                if hasattr(args, "rev") and args.rev:
                    iter_commits_kwargs["rev"] = args.rev
                if hasattr(args, "paths") and args.paths:
                    iter_commits_kwargs["paths"] = args.paths

                repo_changes.all_commits(repo_path, **iter_commits_kwargs)
                print("\nExtracting Refs...")
                build_diff_lists(
                    str(Path(repo_path) / "changes"),
                    directory=args.directory,
                    skip_time=skip_time,
                    continue_on_error=continue_on_error,
                )
            except Exception as e:
                print(f"Error during commit history extraction: {e}")
                if not continue_on_error:
                    raise

    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C)")
        raise
    except Exception as e:
        print(f"Critical error in extract_refs: {e}")
        if not continue_on_error:
            raise
        print("Continuing despite error due to continue-on-error flag")


def validate(args):
    validations = pd.read_csv(args.path)
    validations["correct"] = validations["correct"].apply(
        lambda x: "true" if x == 1 else "false"
    )
    validations = (
        validations.groupby(["commit"]).agg(lambda x: ",".join(x)).reset_index()
    )
    validations["project"] = validations["project"].apply(lambda x: x.split(",")[0])
    validations = validations.to_dict("records")

    from repomanager import repo_changes, repo_utils

    for validation in validations:
        if (
            validation["commit"] == "bf9c26bb128d50ff8369c3bc7fbfc63d066d1ea8"
            or "false" not in validation["correct"]
        ):
            continue

        repo = validation["project"].split("_")
        print(
            "-----------------------------------------------------------------------------------------------------------"
        )
        print(f"Cloning {repo[0]}/{repo[1]}")
        repo_utils.clone_repo(repo[0], repo[1])

        while not Path(f"./Repos/{repo[1]}").exists():
            time.sleep(1)

        path_to_repo = str(Path("./Repos") / repo[1])
        repo_changes.all_commits(path_to_repo, [validation["commit"]])

        while not Path(
            f"./Repos/{repo[1]}/changes/{validation['commit']}.csv"
        ).exists():
            time.sleep(1)

        print(f"Validation of {validation['type']}: {validation['correct']}")

        changes_path = str(Path("./Repos") / repo[1] / "changes")
        build_diff_lists(changes_path, validation["commit"])


# Global cache for AST parsing and tree conversion
_ast_cache = {}
_tree_cache = {}


def get_content_hash(content):
    """Generate hash for content to use as cache key"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def cached_eval_and_tree(file_content):
    """Cache AST evaluation and tree conversion"""
    content_hash = get_content_hash(file_content)

    if content_hash in _tree_cache:
        return _tree_cache[content_hash]

    try:
        # Parse AST only once per unique content
        if content_hash in _ast_cache:
            ast_node = _ast_cache[content_hash]
        else:
            ast_node = eval(file_content)
            _ast_cache[content_hash] = ast_node

        # Convert to tree only once per unique content
        tree = to_tree(ast_node)
        _tree_cache[content_hash] = tree
        return tree
    except Exception:
        # Return None for invalid content
        return None


def populate(row, rev_a, rev_b):
    path = row["Path"]

    # Early exit for identical content
    old_content = row["oldFileContent"]
    new_content = row["currentFileContent"]

    if old_content == new_content:
        return  # No changes, skip processing

    # Use cached parsing
    old_tree = cached_eval_and_tree(old_content)
    new_tree = cached_eval_and_tree(new_content)

    if old_tree is None or new_tree is None:
        return  # Skip invalid files

    rev_a.extract_code_elements(old_tree, path)
    rev_b.extract_code_elements(new_tree, path)


def process_commit_file(args):
    """Process a single commit file - designed for parallel execution"""
    name, file_size, changes_path, directory, skip_time = args

    start_time_commit = time.time()
    commit_hash = name.split(".")[0][:8]

    try:
        df = pd.read_csv(Path(changes_path) / name)
        if directory is not None:
            df = df[df["Path"].isin(directory)]

        rev_a = Rev()
        rev_b = Rev()

        # Process each row (file change) in the commit
        for _, row in df.iterrows():
            populate(row, rev_a, rev_b)

        if skip_time is not None:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(float(skip_time) * 60))

        try:
            rev_difference = rev_a.revision_difference(rev_b)
            refs = rev_difference.get_refactorings()

            # Return results instead of printing directly
            results = []
            for ref in refs:
                results.append((ref, name.split(".")[0]))

            elapsed = time.time() - start_time_commit
            return {
                "success": True,
                "commit_hash": commit_hash,
                "refactorings": results,
                "files_processed": len(df),
                "elapsed_time": elapsed,
            }

        except Exception as e:
            return {
                "success": False,
                "commit_hash": commit_hash,
                "error": str(e),
                "elapsed_time": time.time() - start_time_commit,
            }
        except TimeoutError:
            return {
                "success": False,
                "commit_hash": commit_hash,
                "error": "Timeout",
                "elapsed_time": time.time() - start_time_commit,
            }
        finally:
            if skip_time is not None:
                signal.alarm(0)

    except Exception as e:
        return {
            "success": False,
            "commit_hash": commit_hash,
            "error": f"File read error: {str(e)}",
            "elapsed_time": time.time() - start_time_commit,
        }


def process_commits_parallel(
    csv_files_with_size,
    changes_path,
    directory,
    skip_time,
    max_workers=None,
    continue_on_error=False,
):
    """Process commits in parallel using ThreadPoolExecutor"""
    if max_workers is None:
        max_workers = min(4, len(csv_files_with_size))  # Conservative default

    # Prepare arguments for parallel processing
    args_list = [
        (name, file_size, changes_path, directory, skip_time)
        for name, file_size in csv_files_with_size
    ]

    refactorings = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show progress
        pbar = tqdm(
            desc="Extracting Refs (Parallel)", unit="file", total=len(args_list)
        )

        # Submit all tasks
        future_to_commit = {
            executor.submit(process_commit_file, args): args[0] for args in args_list
        }

        # Process completed tasks
        for future in future_to_commit:
            try:
                result = future.result()
                pbar.update(1)

                if result["success"]:
                    refactorings.extend(result["refactorings"])
                    # Print refactorings as they complete
                    for ref, commit in result["refactorings"]:
                        print(">>>", str(ref))

                    pbar.set_postfix(
                        commit=result["commit_hash"][:8],
                        files=result["files_processed"],
                        time=f"{result['elapsed_time']:.1f}s",
                        refresh=True,
                    )
                else:
                    print(
                        f"Failed to process commit {result['commit_hash']}: {result['error']}"
                    )
            except Exception as e:
                commit_name = future_to_commit[future]
                print(f"Unexpected error processing commit {commit_name}: {e}")
                if not continue_on_error:
                    # Cancel remaining tasks and re-raise
                    for f in future_to_commit:
                        if not f.done():
                            f.cancel()
                    raise

        pbar.close()

    return refactorings


def build_diff_lists_args(args):
    build_diff_lists(args.path, args.commit)
