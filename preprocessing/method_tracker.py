"""
メソッド追跡システム
リファクタリングが発生してもメソッドを一意に追跡する
"""

import json
import uuid
from typing import Dict, List, Optional, Tuple


class MethodTracker:
    """メソッドの追跡とID管理を行うクラス"""

    def __init__(self, refactoring_data_path: str, commit_timestamps_path: str):
        """
        Args:
            refactoring_data_path: リファクタリングデータのJSONファイルパス
            commit_timestamps_path: コミットタイムスタンプのJSONファイルパス
        """
        self.refactoring_data_path = refactoring_data_path
        self.commit_timestamps_path = commit_timestamps_path

        # データ読み込み
        with open(refactoring_data_path, "r", encoding="utf-8") as f:
            self.refactoring_data = json.load(f)

        with open(commit_timestamps_path, "r", encoding="utf-8") as f:
            self.commit_timestamps = json.load(f)

        # 追跡データ
        self.method_tracking: Dict[str, dict] = {}
        self.method_lineage: Dict[str, List[dict]] = {
            "extract_relations": [],
            "inline_relations": [],
        }

        # メソッド識別子(Location/Name)からIDへのマッピング
        self.method_identifier_to_id: Dict[str, str] = {}

    def _generate_method_id(self) -> str:
        """新しいメソッドIDを生成"""
        return f"method_{str(uuid.uuid4())[:8]}"

    def _get_method_identifier(self, location: str, name: str) -> str:
        """メソッドの一意識別子を生成"""
        return f"{location}/{name}"

    def _sort_refactorings_by_time(self) -> List[dict]:
        """リファクタリングデータを時系列でソート"""
        sorted_data = sorted(
            self.refactoring_data,
            key=lambda x: self.commit_timestamps.get(
                x["Commit"], "9999-12-31T00:00:00Z"
            ),
        )
        return sorted_data

    def _get_or_create_method_id(self, location: str, name: str, commit: str) -> str:
        """
        メソッドIDを取得または新規作成

        Args:
            location: メソッドの場所 (例: "example.py/example_class")
            name: メソッド名
            commit: コミットハッシュ

        Returns:
            メソッドID
        """
        identifier = self._get_method_identifier(location, name)

        if identifier in self.method_identifier_to_id:
            return self.method_identifier_to_id[identifier]

        # 新規メソッドIDを作成
        method_id = self._generate_method_id()
        self.method_identifier_to_id[identifier] = method_id

        # 追跡データに初期化
        self.method_tracking[method_id] = {
            "current_name": name,
            "current_location": location,
            "created_at_commit": commit,
            "history": [],
            "lineage": {"parent_methods": [], "child_methods": []},
        }

        return method_id

    def handle_rename_method(self, refactoring: dict) -> None:
        """Rename Methodの処理"""
        old_name = refactoring["Original"]
        new_name = refactoring["Updated"]
        location = refactoring["Location"]
        commit = refactoring["Commit"]

        # 旧名前でIDを取得
        old_identifier = self._get_method_identifier(location, old_name)

        if old_identifier in self.method_identifier_to_id:
            method_id = self.method_identifier_to_id[old_identifier]

            # 新しい識別子に更新
            new_identifier = self._get_method_identifier(location, new_name)
            self.method_identifier_to_id[new_identifier] = method_id
            del self.method_identifier_to_id[old_identifier]

            # 追跡データを更新
            self.method_tracking[method_id]["current_name"] = new_name
            self.method_tracking[method_id]["history"].append(
                {
                    "commit": commit,
                    "timestamp": self.commit_timestamps.get(commit),
                    "refactoring_type": "Rename Method",
                    "old_name": old_name,
                    "new_name": new_name,
                    "location": location,
                }
            )
        else:
            # 既存のメソッドが見つからない場合は新規作成
            method_id = self._get_or_create_method_id(location, new_name, commit)
            self.method_tracking[method_id]["history"].append(
                {
                    "commit": commit,
                    "timestamp": self.commit_timestamps.get(commit),
                    "refactoring_type": "Rename Method",
                    "old_name": old_name,
                    "new_name": new_name,
                    "location": location,
                }
            )

    def handle_extract_method(self, refactoring: dict) -> None:
        """Extract Methodの処理"""
        original_name = refactoring["Original"]
        extracted_name = refactoring["Updated"]
        location = refactoring["Location"]
        commit = refactoring["Commit"]
        extracted_lines = refactoring.get("Extracted/Inlined Lines", [])

        # 元のメソッドIDを取得または作成
        original_id = self._get_or_create_method_id(location, original_name, commit)

        # 抽出されたメソッドの新規IDを作成
        extracted_id = self._get_or_create_method_id(location, extracted_name, commit)

        # 履歴を更新
        self.method_tracking[original_id]["history"].append(
            {
                "commit": commit,
                "timestamp": self.commit_timestamps.get(commit),
                "refactoring_type": "Extract Method",
                "extracted_method": extracted_name,
                "extracted_method_id": extracted_id,
                "extracted_lines": extracted_lines,
            }
        )

        self.method_tracking[extracted_id]["history"].append(
            {
                "commit": commit,
                "timestamp": self.commit_timestamps.get(commit),
                "refactoring_type": "Extracted from Method",
                "parent_method": original_name,
                "parent_method_id": original_id,
                "extracted_lines": extracted_lines,
            }
        )

        # 系譜を更新
        self.method_tracking[original_id]["lineage"]["child_methods"].append(
            extracted_id
        )
        self.method_tracking[extracted_id]["lineage"]["parent_methods"].append(
            original_id
        )

        # 系譜データに追加
        self.method_lineage["extract_relations"].append(
            {
                "parent_method_id": original_id,
                "parent_method_name": original_name,
                "child_method_id": extracted_id,
                "child_method_name": extracted_name,
                "commit": commit,
                "timestamp": self.commit_timestamps.get(commit),
                "extracted_lines": extracted_lines,
            }
        )

    def handle_inline_method(self, refactoring: dict) -> None:
        """Inline Methodの処理"""
        target_name = refactoring["Original"]
        inlined_name = refactoring["Updated"]
        location = refactoring["Location"]
        commit = refactoring["Commit"]
        inlined_lines = refactoring.get("Extracted/Inlined Lines", [])

        # 両方のメソッドIDを取得
        target_id = self._get_or_create_method_id(location, target_name, commit)

        inlined_identifier = self._get_method_identifier(location, inlined_name)
        if inlined_identifier in self.method_identifier_to_id:
            inlined_id = self.method_identifier_to_id[inlined_identifier]

            # インライン化されたメソッドの履歴を更新
            self.method_tracking[inlined_id]["history"].append(
                {
                    "commit": commit,
                    "timestamp": self.commit_timestamps.get(commit),
                    "refactoring_type": "Inlined into Method",
                    "target_method": target_name,
                    "target_method_id": target_id,
                    "inlined_lines": inlined_lines,
                }
            )

            # ターゲットメソッドの履歴を更新
            self.method_tracking[target_id]["history"].append(
                {
                    "commit": commit,
                    "timestamp": self.commit_timestamps.get(commit),
                    "refactoring_type": "Inline Method",
                    "inlined_method": inlined_name,
                    "inlined_method_id": inlined_id,
                    "inlined_lines": inlined_lines,
                }
            )

            # 系譜データに追加
            self.method_lineage["inline_relations"].append(
                {
                    "inlined_method_id": inlined_id,
                    "inlined_method_name": inlined_name,
                    "target_method_id": target_id,
                    "target_method_name": target_name,
                    "commit": commit,
                    "timestamp": self.commit_timestamps.get(commit),
                    "inlined_lines": inlined_lines,
                }
            )

            # インライン化されたメソッドは削除されるので、識別子マッピングから削除
            del self.method_identifier_to_id[inlined_identifier]
        else:
            # 既存のメソッドが見つからない場合も記録
            self.method_tracking[target_id]["history"].append(
                {
                    "commit": commit,
                    "timestamp": self.commit_timestamps.get(commit),
                    "refactoring_type": "Inline Method",
                    "inlined_method": inlined_name,
                    "inlined_lines": inlined_lines,
                    "note": "Inlined method ID not found",
                }
            )

    def handle_push_down_method(self, refactoring: dict) -> None:
        """Push Down Methodの処理"""
        method_name = refactoring["Original"]
        old_location = refactoring["Old Location"]
        new_location = refactoring["New Location"]
        commit = refactoring["Commit"]

        # 旧Locationでメソッドを検索
        old_identifier = self._get_method_identifier(old_location, method_name)

        if old_identifier in self.method_identifier_to_id:
            method_id = self.method_identifier_to_id[old_identifier]

            # 新しいLocationに更新
            new_identifier = self._get_method_identifier(new_location, method_name)
            self.method_identifier_to_id[new_identifier] = method_id
            del self.method_identifier_to_id[old_identifier]

            # 追跡データを更新
            self.method_tracking[method_id]["current_location"] = new_location
            self.method_tracking[method_id]["history"].append(
                {
                    "commit": commit,
                    "timestamp": self.commit_timestamps.get(commit),
                    "refactoring_type": "Push Down Method",
                    "old_location": old_location,
                    "new_location": new_location,
                    "method_name": method_name,
                }
            )
        else:
            # 既存のメソッドが見つからない場合は新規作成
            method_id = self._get_or_create_method_id(new_location, method_name, commit)
            self.method_tracking[method_id]["history"].append(
                {
                    "commit": commit,
                    "timestamp": self.commit_timestamps.get(commit),
                    "refactoring_type": "Push Down Method",
                    "old_location": old_location,
                    "new_location": new_location,
                    "method_name": method_name,
                }
            )

    def handle_parameter_change(self, refactoring: dict, refactoring_type: str) -> None:
        """パラメータ変更(Add/Remove/Change)の処理"""
        method_name = refactoring["Original"]
        location = refactoring["Location"]
        commit = refactoring["Commit"]

        # メソッドIDを取得または作成
        method_id = self._get_or_create_method_id(location, method_name, commit)

        # 履歴を更新
        history_entry = {
            "commit": commit,
            "timestamp": self.commit_timestamps.get(commit),
            "refactoring_type": refactoring_type,
            "description": refactoring.get("Description", []),
        }

        self.method_tracking[method_id]["history"].append(history_entry)

    def handle_rename_class(self, refactoring: dict) -> None:
        """Rename Classの処理 - 該当クラス内の全メソッドのLocationを更新"""
        old_class = refactoring["Original"]
        new_class = refactoring["Updated"]
        module = refactoring["Location"]
        commit = refactoring["Commit"]

        old_location_prefix = f"{module}/{old_class}"
        new_location_prefix = f"{module}/{new_class}"

        # 該当クラス内の全メソッドを更新
        identifiers_to_update = []
        for identifier, method_id in self.method_identifier_to_id.items():
            if identifier.startswith(old_location_prefix):
                identifiers_to_update.append((identifier, method_id))

        for old_identifier, method_id in identifiers_to_update:
            # 新しい識別子を作成
            new_identifier = old_identifier.replace(
                old_location_prefix, new_location_prefix, 1
            )

            # マッピングを更新
            self.method_identifier_to_id[new_identifier] = method_id
            del self.method_identifier_to_id[old_identifier]

            # Locationを更新
            new_location = self.method_tracking[method_id]["current_location"].replace(
                old_location_prefix, new_location_prefix, 1
            )
            self.method_tracking[method_id]["current_location"] = new_location

            # 履歴を追加
            self.method_tracking[method_id]["history"].append(
                {
                    "commit": commit,
                    "timestamp": self.commit_timestamps.get(commit),
                    "refactoring_type": "Class Renamed",
                    "old_class": old_class,
                    "new_class": new_class,
                    "module": module,
                }
            )

    def build_tracking_data(self) -> None:
        """リファクタリングデータからメソッド追跡データを構築"""
        # 時系列でソート
        sorted_refactorings = self._sort_refactorings_by_time()

        # 各リファクタリングを処理
        for refactoring in sorted_refactorings:
            refactoring_type = refactoring.get("Refactoring Type")

            # リストの場合は最初の要素を使用
            if isinstance(refactoring_type, list):
                refactoring_type = refactoring_type[0] if refactoring_type else None

            try:
                if refactoring_type == "Rename Method":
                    self.handle_rename_method(refactoring)
                elif refactoring_type == "Extract Method":
                    self.handle_extract_method(refactoring)
                elif refactoring_type == "Inline Method":
                    self.handle_inline_method(refactoring)
                elif refactoring_type == "Push Down Method":
                    self.handle_push_down_method(refactoring)
                elif refactoring_type == "Add Parameter":
                    self.handle_parameter_change(refactoring, "Add Parameter")
                elif refactoring_type == "Remove Parameter":
                    self.handle_parameter_change(refactoring, "Remove Parameter")
                elif refactoring_type == "Change/Rename Parameter":
                    self.handle_parameter_change(refactoring, "Change/Rename Parameter")
                elif refactoring_type == "Rename Class":
                    self.handle_rename_class(refactoring)
                else:
                    print(f"Warning: Unknown refactoring type: {refactoring_type}")
            except Exception as e:
                print(f"Error processing refactoring: {refactoring}")
                print(f"Error: {e}")
                raise

    def save_tracking_data(self, output_dir: str) -> Tuple[str, str]:
        """
        追跡データをJSON形式で保存

        Args:
            output_dir: 出力ディレクトリ

        Returns:
            (method_tracking_path, method_lineage_path)
        """
        import os

        # ディレクトリが存在しない場合は作成
        os.makedirs(output_dir, exist_ok=True)

        tracking_path = os.path.join(output_dir, "method_tracking.json")
        lineage_path = os.path.join(output_dir, "method_lineage.json")

        with open(tracking_path, "w", encoding="utf-8") as f:
            json.dump(self.method_tracking, f, indent=2, ensure_ascii=False)

        with open(lineage_path, "w", encoding="utf-8") as f:
            json.dump(self.method_lineage, f, indent=2, ensure_ascii=False)

        return tracking_path, lineage_path

    def get_method_info(self, method_id: str) -> Optional[dict]:
        """指定されたメソッドIDの情報を取得"""
        return self.method_tracking.get(method_id)

    def find_method_by_name_and_location(
        self, name: str, location: str
    ) -> Optional[str]:
        """メソッド名とLocationからメソッドIDを検索"""
        identifier = self._get_method_identifier(location, name)
        return self.method_identifier_to_id.get(identifier)

    def get_statistics(self) -> dict:
        """追跡統計情報を取得"""
        total_methods = len(self.method_tracking)
        total_extracts = len(self.method_lineage["extract_relations"])
        total_inlines = len(self.method_lineage["inline_relations"])

        refactoring_counts = {}
        for method_id, data in self.method_tracking.items():
            for history_entry in data["history"]:
                ref_type = history_entry.get("refactoring_type", "Unknown")
                refactoring_counts[ref_type] = refactoring_counts.get(ref_type, 0) + 1

        return {
            "total_methods": total_methods,
            "total_extract_relations": total_extracts,
            "total_inline_relations": total_inlines,
            "refactoring_counts": refactoring_counts,
        }


def main():
    """メイン処理"""
    import argparse
    import os

    # コマンドライン引数をパース
    parser = argparse.ArgumentParser(
        description="メソッド追跡システム - リファクタリングが発生してもメソッドを一意に追跡する"
    )
    parser.add_argument(
        "--repo",
        "-r",
        default="DummyRef",
        help="リポジトリ名（デフォルト: DummyRef）",
    )
    parser.add_argument(
        "--refactoring-data",
        help="リファクタリングデータのJSONファイルパス（指定しない場合は data/{repo}/refactoring_mining.json を使用）",
    )
    parser.add_argument(
        "--commit-timestamps",
        help="コミットタイムスタンプのJSONファイルパス（指定しない場合は data/{repo}/commit_timestamps.json を使用）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="出力ディレクトリ（指定しない場合は data/{repo} を使用）",
    )

    args = parser.parse_args()

    # パス設定
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # リファクタリングデータのパス
    if args.refactoring_data:
        refactoring_data_path = args.refactoring_data
    else:
        refactoring_data_path = os.path.join(
            base_dir, "data", args.repo, "refactoring_mining.json"
        )

    # コミットタイムスタンプのパス
    if args.commit_timestamps:
        commit_timestamps_path = args.commit_timestamps
    else:
        commit_timestamps_path = os.path.join(
            base_dir, "data", args.repo, "commit_timestamps.json"
        )

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(base_dir, "data", args.repo)

    # ファイルの存在確認
    if not os.path.exists(refactoring_data_path):
        print(f"Error: Refactoring data file not found: {refactoring_data_path}")
        return

    if not os.path.exists(commit_timestamps_path):
        print(f"Error: Commit timestamps file not found: {commit_timestamps_path}")
        return

    print(f"Repository: {args.repo}")
    print(f"Refactoring data: {refactoring_data_path}")
    print(f"Commit timestamps: {commit_timestamps_path}")
    print(f"Output directory: {output_dir}")
    print()

    # トラッカーを初期化
    tracker = MethodTracker(refactoring_data_path, commit_timestamps_path)

    # 追跡データを構築
    print("Building method tracking data...")
    tracker.build_tracking_data()

    # 統計情報を表示
    stats = tracker.get_statistics()
    print("\n=== Statistics ===")
    print(f"Total tracked methods: {stats['total_methods']}")
    print(f"Total extract relations: {stats['total_extract_relations']}")
    print(f"Total inline relations: {stats['total_inline_relations']}")
    print("\nRefactoring counts:")
    for ref_type, count in sorted(stats["refactoring_counts"].items()):
        print(f"  {ref_type}: {count}")

    # データを保存
    print("\nSaving tracking data...")
    tracking_path, lineage_path = tracker.save_tracking_data(output_dir)
    print(f"Method tracking data saved to: {tracking_path}")
    print(f"Method lineage data saved to: {lineage_path}")


if __name__ == "__main__":
    main()
