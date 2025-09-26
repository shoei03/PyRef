from pathlib import Path

import git
from git import Repo


def clone_repo(username, repo_name):
    git_url = "https://github.com/" + username + "/" + repo_name + ".git"
    repo_path = Path("./Repos/" + repo_name).resolve()
    if repo_path.exists():
        print("Repo Already Cloned.")
        return repo_path
    try:
        print("Cloning Repo...")

        repo = Repo.clone_from(git_url, str(repo_path), branch="main")
        print(
            f"Successfully cloned {username}/{repo_name} from main branch to {repo_path}"
        )
        return repo_path
    except git.exc.GitCommandError:
        repo = Repo.clone_from(git_url, str(repo_path), branch="master")
        print(
            f"Successfully cloned {username}/{repo_name} from master branch to {repo_path}"
        )
        return repo_path


def clone_repo_args(args):
    clone_repo(args.username, args.reponame)
