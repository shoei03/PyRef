import json
import os
import signal
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


def filter_refactorings_by_method(refactorings, method_name, match_mode="exact"):
    """
    Filter refactorings by method name based on different matching modes.

    Args:
        refactorings: List of (refactoring_object, commit_hash) tuples
        method_name: Target method name to filter by
        match_mode: 'exact', 'partial', or 'regex'

    Returns:
        Filtered list of refactorings related to the specified method
    """
    if not method_name:
        return refactorings

    filtered_refactorings = []

    for ref, commit in refactorings:
        if is_method_related_refactoring(ref, method_name, match_mode):
            filtered_refactorings.append((ref, commit))

    return filtered_refactorings


def is_method_related_refactoring(refactoring, method_name, match_mode):
    """
    Check if a refactoring is related to the specified method.

    Args:
        refactoring: Refactoring object (RenameRef, ExtractInlineRef, MoveRef, etc.)
        method_name: Target method name
        match_mode: 'exact', 'partial', or 'regex'

    Returns:
        Boolean indicating if the refactoring is related to the method
    """
    # Get method names involved in the refactoring
    involved_methods = get_involved_method_names(refactoring)

    # Check if any involved method matches the target
    for method in involved_methods:
        if matches_method_name(method, method_name, match_mode):
            return True

    return False


def get_involved_method_names(refactoring):
    """
    Extract all method names involved in a refactoring.

    Args:
        refactoring: Refactoring object

    Returns:
        List of method names involved in the refactoring
    """
    method_names = []

    # All refactoring types have _from and _to attributes
    if hasattr(refactoring, "_from") and refactoring._from:
        method_names.append(refactoring._from)

    if hasattr(refactoring, "_to") and refactoring._to:
        method_names.append(refactoring._to)

    # For some refactorings, we might want to include class context
    if hasattr(refactoring, "_removed_m") and refactoring._removed_m:
        if hasattr(refactoring._removed_m, "name"):
            method_names.append(refactoring._removed_m.name)

    if hasattr(refactoring, "_added_m") and refactoring._added_m:
        if hasattr(refactoring._added_m, "name"):
            method_names.append(refactoring._added_m.name)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(method_names))


def matches_method_name(method, target_method, match_mode):
    """
    Check if a method name matches the target based on the matching mode.

    Args:
        method: Method name to check
        target_method: Target method name
        match_mode: 'exact', 'partial', or 'regex'

    Returns:
        Boolean indicating if there's a match
    """
    if not method or not target_method:
        return False

    if match_mode == "exact":
        return method == target_method
    elif match_mode == "partial":
        return target_method in method or method in target_method
    elif match_mode == "regex":
        try:
            import re

            return bool(re.search(target_method, method))
        except re.error:
            # If regex is invalid, fall back to exact match
            return method == target_method

    return False


def build_diff_lists(
    changes_path,
    commit=None,
    directory=None,
    skip_time=None,
    method_name=None,
    match_mode="exact",
):
    refactorings = []
    t0 = time.time()
    if commit is not None:
        print(commit)
        name = commit + ".csv"
        import pandas as pd

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
    else:
        for root, dirs, files in os.walk(changes_path):
            for ind, name in enumerate(files):
                if name.endswith(".csv"):
                    print(ind, "/", len(files), " --- ", name[:-4])
                    import pandas as pd

                    df = pd.read_csv(changes_path + "/" + name)
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

    # Apply method filtering if specified
    if method_name:
        original_count = len(refactorings)
        refactorings = filter_refactorings_by_method(
            refactorings, method_name, match_mode
        )
        filtered_count = len(refactorings)
        print(
            f"Filtered refactorings by method '{method_name}' ({match_mode} match): {filtered_count}/{original_count}"
        )

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
        print("commit: %3s - %s" % (ref[1], str(ref[0]).strip()))
        data = ref[0].to_json_format()
        data["Commit"] = ref[1]
        json_outputs.append(data)
        # ref[0].to_graph()
    changes_path = changes_path.replace("//", "/")
    repo_name = changes_path.split("/")[-3]

    # Create output filename based on filtering
    if method_name:
        output_filename = f"{repo_name}_{method_name}_{match_mode}_data.json"
        # Create enhanced output format for method tracking
        enhanced_output = {
            "target_method": method_name,
            "match_mode": match_mode,
            "refactoring_history": json_outputs,
            "summary": {
                "total_refactorings": len(json_outputs),
                "refactoring_types": {},
            },
        }

        # Count refactoring types
        for ref_data in json_outputs:
            ref_type = ref_data.get("Refactoring Type", "Unknown")
            enhanced_output["summary"]["refactoring_types"][ref_type] = (
                enhanced_output["summary"]["refactoring_types"].get(ref_type, 0) + 1
            )

        with open(output_filename, "w") as outfile:
            outfile.write(json.dumps(enhanced_output, indent=4))
    else:
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
    if args.skip is not None:
        skip_time = args.skip
        print(
            "\nCommit will be skipped if the processing time is longer than",
            skip_time,
            "minutes.",
        )
    else:
        skip_time = None

    # Get method filtering parameters
    method_name = getattr(args, "method", None)
    match_mode = getattr(args, "match_mode", "exact")

    if method_name:
        print(f"\nFiltering by method: '{method_name}' (match mode: {match_mode})")

    if args.commit is not None:
        repo_changes.all_commits(repo_path, [args.commit])
        print("\nExtracting Refs...")
        build_diff_lists(
            repo_path + "/changes/",
            args.commit,
            args.directory,
            skip_time,
            method_name,
            match_mode,
        )
    else:
        print("\nExtracting commit history...")
        repo_changes.all_commits(repo_path)
        print("\nExtracting Refs...")
        build_diff_lists(
            repo_path + "/changes/",
            directory=args.directory,
            skip_time=skip_time,
            method_name=method_name,
            match_mode=match_mode,
        )


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
