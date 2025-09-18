# PyRef

## Description

PyRef is a tool that automatically detect mainly method-level refactoring operations in Python projects.

Current supported refactoring operations:

- Rename Method
- Add Parameter
- Remove Parameter
- Change/Rename Parameter
- Extract Method
- Inline Method
- Move Method
- Pull Up Method
- Push Down Method

## Usage

Clone a repository from GitHub using PyRef:

```sh
python3 main.py repoClone -u "username" -r "Repo Name"
```

You can also use git command to clone the repository.

Extract refactorings from a given repository

```sh
python3 main.py getrefs -r "[PATH_TO_REPOSITORY]"
```

You can also use flag _-s_ to skip the commit which takes more than N minutes to extract the refactorings. For example, the following command skips commits which were processed for more than 10 minutes:

```sh
python3 main.py getrefs -r "[PATH_TO_REPOSITORY]" -s 10
```

If you want to look into specific commit, you can use flag _-c_.
If you want to look into specific directory, you can use flag _-d_.
If you want to track refactorings for a specific method, you can use flag _-m_.

```sh
python3 main.py getrefs -r "[PATH_TO_REPOSITORY]" -c "[CommitHash]" -d "[Directory]"
```

Extract refactorings for a specific method:

```sh
python3 main.py getrefs -r "[PATH_TO_REPOSITORY]" -m "[MethodName]"
```

You can also use _--match-mode_ to change the matching behavior:

- _exact_: exact method name match (default)
- _partial_: partial method name match
- _regex_: regular expression match

```sh
python3 main.py getrefs -r "[PATH_TO_REPOSITORY]" -m "calc" --match-mode partial
```

The detected refactorings will be recorded in the current folder as a json file "[project]_data.json".
When using the *-m* flag, results will be saved as "[project]_[method]\_[match_mode]\_data.json" with additional method tracking information.

## Play with PyRef

You will need to first install the third-party dependencies. You can use the following command in the folder of PyRef:

```sh
pip3 install -r requirements.txt
```

**Note: Pandas of a version lower than 2.0.0 is required, as the newer versions of pandas changed ".append" (used in the PyRef code) to ".\_append" to avoid confusion with ".append" in Python (Thanks to Zhi Li for pointing this out).**

We provide a toy project for you to test PyRef, which can be found at https://github.com/PyRef/DummyRef
Please execute the following commands in order:

```sh
python3 main.py repoClone -u "PyRef" -r "DummyRef"
python3 main.py getrefs -r "Repos/DummyRef"
```

The detected refactorings can be found in the file "DummyRef_data.json"

## Dataset for the Paper

This tool was part of the following study:

H. Atwi, B. Lin, N. Tsantalis, Y. Kashiwa, Y. Kamei, N. Ubayashi, G. Bavota and M. Lanza, "PyRef: Refactoring Detection in Python Projects," 2021 IEEE 21st International Working Conference on Source Code Analysis and Manipulation (SCAM), 2021, accepted.

The labeled oracle used in the paper can be found in the file "data/dataset.csv".
