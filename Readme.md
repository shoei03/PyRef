# PyRef

PyRef is a tool that automatically detects mainly method-level refactoring operations in Python projects using Docker for easy deployment and consistent environments.

## Supported Refactoring Operations

- Rename Method
- Add Parameter
- Remove Parameter
- Change/Rename Parameter
- Extract Method
- Inline Method
- Move Method
- Pull Up Method
- Push Down Method

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed on your system

### 1. Setup

```sh
# Clone this repository
git clone https://github.com/shoei03/PyRef.git
cd PyRef

# Build the Docker image
docker-compose build
```

### 2. Basic Usage

```sh
# Show available commands
docker-compose run --rm pyref --help

# Clone a repository for analysis
docker-compose run --rm pyref repoClone -u "PyRef" -r "DummyRef"

# Analyze refactorings in the repository
docker-compose run --rm pyref getrefs -r "Repos/DummyRef"
```

### 3. View Results

Results are saved as JSON files in the project root (e.g., `DummyRef_data.json`)

## Available Commands

### Repository Management

```sh
# Clone a GitHub repository
docker-compose run --rm pyref repoClone -u "username" -r "repository_name"
```

### Refactoring Analysis

```sh
# Basic analysis
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]"

# Skip commits that take more than N minutes
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" -s 10

# Analyze specific commit
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" -c "[COMMIT_HASH]"

# Analyze specific directory
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" -d "[DIRECTORY_PATH]"
```

### Error Handling Options

```sh
# Continue processing even if errors occur (recommended for large repositories)
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" --continue-on-error

# Combine with other options for robust analysis
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" \
  --since "2023-01-01" \
  --until "2023-12-31" \
  --continue-on-error \
  -s 10
```

## Advanced Usage Examples

### Performance Optimization

```sh
# Process only the latest 10 commits
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" --max-count 10

# Skip the first 5 commits
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" --skip-commits 5

# Process commits from the last week
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" --since "1 week ago"
```

### Targeted Analysis

```sh
# Analyze commits until a specific date
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" --until "2023-12-31"

# Start from a specific branch or tag
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" --rev "develop"

# Focus on specific file paths
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" --paths "src/*.py" "tests/*.py"
```

### Complex Analysis Scenarios

```sh
# Latest 50 commits, specific directory
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" \
  --max-count 50 \
  --paths "src/core/*.py"

# Date range analysis
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" \
  --since "2023-01-01" \
  --until "2023-12-31"

# Continue processing even if errors occur
docker-compose run --rm pyref getrefs -r "Repos/[REPO_NAME]" \
  --continue-on-error
```

## Development Environment

For interactive development and debugging:

```sh
# Start development container
docker-compose up -d pyref-dev

# Access interactive shell
docker-compose exec pyref-dev bash

# Stop development container
docker-compose down
```

## Alternative: Direct Docker Usage

If you prefer not to use Docker Compose:

```sh
# Build image
docker build -t pyref .

# Run commands
docker run --rm -v $(pwd):/app pyref [COMMAND] [ARGS]

# Example
docker run --rm -v $(pwd):/app pyref getrefs -r "Repos/DummyRef"
```

## Local Installation (Alternative)

If you prefer to run without Docker:

```sh
pip3 install -r requirements.txt
python3 main.py [COMMAND] [ARGS]
```

**Note**: Requires Python 3.9+ and pandas < 2.0.0 for compatibility.

## Output Format

Results are saved as JSON files with the following naming convention:

- Basic analysis: `[project_name]_data.json`

Example output structure:

```json
[
  {
    "Refactoring Type": ["Add Parameter"],
    "Original": "methodName",
    "Updated": "methodName",
    "Location": "file.py/ClassName",
    "Original Line": 10,
    "Updated Line": 10,
    "Description": ["The parameters [param1] are added to the method..."],
    "Commit": "abcd1234..."
  }
]
```

## Docker Environment Details

- **Base Image**: Python 3.9-slim (optimized for compatibility with pandas 1.2.2)
- **Security**: Runs as non-root user `pyref`
- **Persistence**:
  - Repositories stored in `./Repos/` directory
  - Results saved to project root
  - Data files mounted to `./data/`
- **Dependencies**: Pre-installed and version-locked for consistency

## Academic Reference

This tool was developed as part of the following research:

H. Atwi, B. Lin, N. Tsantalis, Y. Kashiwa, Y. Kamei, N. Ubayashi, G. Bavota and M. Lanza, "PyRef: Refactoring Detection in Python Projects," 2021 IEEE 21st International Working Conference on Source Code Analysis and Manipulation (SCAM), 2021.

The labeled oracle dataset is available in `data/dataset.csv`.
