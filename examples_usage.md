# PyRef 拡張機能の使用例

## iter_commits オプションの使用例

### 1. 最新の 10 コミットのみを処理

```bash
python3 main.py getrefs -r "/path/to/repo" --max-count 10
```

### 2. 最初の 5 コミットをスキップして処理

```bash
python3 main.py getrefs -r "/path/to/repo" --skip-commits 5
```

### 3. 特定の日付以降のコミットのみを処理

```bash
python3 main.py getrefs -r "/path/to/repo" --since "2023-01-01"
python3 main.py getrefs -r "/path/to/repo" --since "1 week ago"
```

### 4. 特定の日付までのコミットのみを処理

```bash
python3 main.py getrefs -r "/path/to/repo" --until "2023-12-31"
python3 main.py getrefs -r "/path/to/repo" --until "1 day ago"
```

### 5. 特定のブランチまたはリビジョンから開始

```bash
python3 main.py getrefs -r "/path/to/repo" --rev "develop"
python3 main.py getrefs -r "/path/to/repo" --rev "v1.0.0"
```

### 6. 特定のファイルパスの変更のみを追跡

```bash
python3 main.py getrefs -r "/path/to/repo" --paths "src/*.py" "tests/*.py"
```

### 7. 組み合わせ使用例

```bash
# 最新50コミット、特定のディレクトリ、特定のメソッドを追跡
python3 main.py getrefs -r "/path/to/repo" --max-count 50 --paths "src/core/*.py" -m "calculate" --match-mode partial

# 1週間前以降、最大20コミット、特定のメソッド
python3 main.py getrefs -r "/path/to/repo" --since "1 week ago" --max-count 20 -m "init" --match-mode exact
```

## repoChanges コマンドでも同様のオプションが使用可能

```bash
python3 main.py repoChanges -p "/path/to/repo" --allcommits --max-count 10
python3 main.py repoChanges -p "/path/to/repo" --allcommits --since "2023-01-01" --until "2023-12-31"
```

## 主な用途

1. **パフォーマンス向上**: 大きなリポジトリで処理時間を短縮
2. **特定期間の分析**: プロジェクトの特定の開発期間に焦点
3. **漸進的分析**: 段階的にコミットを処理
4. **特定ファイルの追跡**: 重要なファイルの変更のみを分析
5. **ブランチ別分析**: 異なるブランチの開発パターンを比較
