import json
import os
import signal
import sys
import threading
import time
from ast import *
from os import path

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


def load_commits_from_file(file_path):
    """
    Load commit hashes from a JSON file.

    Args:
        file_path: Path to the JSON file with commit hashes as keys

    Returns:
        List of commit hashes (keys from the JSON)

    Raises:
        FileNotFoundError: If the file does not exist
        SystemExit: If the file is empty or invalid JSON
    """
    if not os.path.exists(file_path):
        print(f"Error: Commit file not found: {file_path}")
        sys.exit(1)

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print(f"Error: Commit file must be a JSON object: {file_path}")
            sys.exit(1)

        commits = list(data.keys())

        if not commits:
            print(f"Error: Commit file contains no commit hashes: {file_path}")
            sys.exit(1)

        print(f"Loaded {len(commits)} commit(s) from {file_path}")
        return commits

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}: {e}")
        sys.exit(1)


def load_and_sort_commits(repo_path, specific_commits=None):
    """
    Load commit_timestamps.json and return commits sorted in chronological order.

    Args:
        repo_path: Path to the repository (e.g., "Repos/DummyRef")
        specific_commits: List of specific commits to filter and sort.
                         If None, all commits from the file will be sorted.

    Returns:
        List of commit hashes sorted in chronological order (oldest to newest)

    Raises:
        SystemExit: If commit_timestamps.json does not exist
    """
    # Extract repo name from path like "Repos/DummyRef" or "Repos/DummyRef/"
    repo_name = repo_path.rstrip("/").split("/")[-1]
    timestamp_file = os.path.join("data", repo_name, "commit_timestamps.json")

    # Check if timestamp file exists
    if not os.path.exists(timestamp_file):
        print(f"Error: commit_timestamps.json not found: {timestamp_file}")
        print("Please generate it first using: scripts/generate_commit_timestamps.sh")
        sys.exit(1)

    # Load timestamps
    try:
        with open(timestamp_file, "r") as f:
            timestamps = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {timestamp_file}: {e}")
        sys.exit(1)

    # Filter to specific commits if provided
    if specific_commits:
        filtered = {k: v for k, v in timestamps.items() if k in specific_commits}
        if len(filtered) < len(specific_commits):
            missing = set(specific_commits) - set(filtered.keys())
            print(
                f"Warning: {len(missing)} commit(s) not found in timestamps, skipping: {list(missing)[:3]}..."
                if len(missing) > 3
                else f"Warning: {len(missing)} commit(s) not found in timestamps: {missing}"
            )
    else:
        filtered = timestamps

    # Sort by timestamp (ascending = oldest to newest)
    sorted_commits = sorted(filtered.keys(), key=lambda x: filtered[x])

    print(f"Loaded and sorted {len(sorted_commits)} commit(s) in chronological order")
    return sorted_commits


def build_diff_lists(
    changes_path, commit=None, directory=None, skip_time=None, commits=None
):
    refactorings = []
    t0 = time.time()

    # Handle single commit (existing behavior - no sorting needed)
    if commit is not None:
        print(commit)
        name = commit + ".csv"
        df = pd.read_csv(changes_path + "/" + name)
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
            print(">>>", str(ref))
    # Handle multiple commits or all commits - sort chronologically
    else:
        # Extract repo_path from changes_path
        # e.g., "Repos/DummyRef/changes/" -> "Repos/DummyRef"
        repo_path = changes_path.rstrip("/").replace("/changes", "")

        # Load and sort commits chronologically
        sorted_commits = load_and_sort_commits(repo_path, commits)

        print(f"\nProcessing {len(sorted_commits)} commits in chronological order...")

        # Process commits in chronological order
        for idx, commit_hash in enumerate(sorted_commits, 1):
            print(f"\n[{idx}/{len(sorted_commits)}] Processing commit: {commit_hash}")
            name = commit_hash + ".csv"
            csv_path = os.path.join(changes_path, name)

            # Check if CSV exists
            if not os.path.exists(csv_path):
                print(f"Warning: CSV file not found: {csv_path}, skipping")
                continue

            try:
                df = pd.read_csv(csv_path)
                if directory is not None:
                    df = df[df["Path"].isin(directory)]
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
                        print(">>>", str(ref))
                except Exception as e:
                    print("Failed to process commit.", e)
                except TimeoutError:
                    print("Commit skipped due to the long processing time")
                finally:
                    rt.stop()
                    if skip_time is not None:
                        signal.alarm(0)
            except Exception as e:
                print(f"Error reading CSV file {csv_path}: {e}")

    t1 = time.time()
    total = t1 - t0
    print(
        "-----------------------------------------------------------------------------------------------------------"
    )
    print("Total Time:", total)
    print("Total Number of Refactorings:", len(refactorings))
    # No need to sort - refactorings are already in chronological order
    json_outputs = []
    for ref in refactorings:
        print("commit: %3s - %s" % (ref[1], str(ref[0]).strip()))
        data = ref[0].to_json_format()
        data["Commit"] = ref[1]
        json_outputs.append(data)
        # ref[0].to_graph()
    changes_path = changes_path.replace("//", "/")
    # Extract repo name from path like "Repos/DummyRef/changes/" -> "DummyRef"
    path_parts = changes_path.rstrip("/").split("/")
    if "changes" in path_parts:
        # Get the directory before "changes"
        changes_index = path_parts.index("changes")
        repo_name = path_parts[changes_index - 1]
    else:
        # Fallback to previous logic
        repo_name = path_parts[-2] if len(path_parts) >= 2 else "unknown"

    # Create data/{repo_name} directory if it doesn't exist
    output_dir = os.path.join("data", repo_name)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "refactoring_mining.json")
    with open(output_path, "w") as outfile:
        outfile.write(json.dumps(json_outputs, indent=4))

    return refactorings


def extract_refs(args):
    # owner_name = args.repo.split("/")[0]
    # repo_name = args.repo.split("/")[1]

    from repomanager import repo_changes

    repo_path = args.repopath

    # Check for mutually exclusive arguments
    if args.commit is not None and args.commit_file is not None:
        print("Error: Cannot specify both -c/--commit and --commit-file")
        sys.exit(1)

    if args.skip is not None:
        skip_time = args.skip
        print(
            "\nCommit will be skipped if the processing time is longer than",
            skip_time,
            "minutes.",
        )
    else:
        skip_time = None

    # Determine commits to process
    commits_to_process = None

    if args.commit_file is not None:
        # Load commits from file
        commits_to_process = load_commits_from_file(args.commit_file)
        print(f"\nExtracting commit history for {len(commits_to_process)} commit(s)...")
        repo_changes.all_commits(repo_path, commits_to_process)
        print("\nExtracting Refs...")
        # Process all commits in one call
        build_diff_lists(
            repo_path + "/changes/",
            directory=args.directory,
            skip_time=skip_time,
            commits=commits_to_process,
        )
    elif args.commit is not None:
        # Single commit
        repo_changes.all_commits(repo_path, [args.commit])
        print("\nExtracting Refs...")
        build_diff_lists(
            repo_path + "/changes/", args.commit, args.directory, skip_time
        )
    else:
        # All commits
        print("\nExtracting commit history...")
        repo_changes.all_commits(repo_path)
        print("\nExtracting Refs...")
        build_diff_lists(repo_path + "/changes/", args.directory, skip_time=skip_time)


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
        print("Cloning %s/%s" % (repo[0], repo[1]))
        repo_utils.clone_repo(repo[0], repo[1])

        while not path.exists("./Repos/" + repo[1]):
            time.sleep(1)

        path_to_repo = "./Repos/" + repo[1]
        repo_changes.all_commits(path_to_repo, [validation["commit"]])

        while not path.exists(
            "./Repos/" + repo[1] + "/changes/" + validation["commit"] + ".csv"
        ):
            time.sleep(1)

        print("Validation of %s: %s" % (validation["type"], validation["correct"]))

        changes_path = "./Repos/" + repo[1] + "/changes/"
        build_diff_lists(changes_path, validation["commit"])


def populate(row, rev_a, rev_b):
    path = row["Path"]
    rav_a_tree = to_tree(eval(row["oldFileContent"]))
    rev_b_tree = to_tree(eval(row["currentFileContent"]))
    rev_a.extract_code_elements(rav_a_tree, path)
    rev_b.extract_code_elements(rev_b_tree, path)


def build_diff_lists_args(args):
    build_diff_lists(args.path, args.commit)
