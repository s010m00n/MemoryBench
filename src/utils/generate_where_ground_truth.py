"""
生成 where_ground_truth 参数的脚本

该脚本会：
1. 读取所有 locomo*.json 文件
2. 为每个 QA 根据 evidence 字段生成 where_ground_truth（包含所有涉及的 session id）
3. 将更新后的数据写回文件
"""

import json
import re
from pathlib import Path
from typing import List, Set


def extract_session_ids_from_evidence(evidence: List[str]) -> List[int]:
    """
    从 evidence 列表中提取所有涉及的 session id

    Args:
        evidence: 格式如 ["D1:3", "D2:8", "D1:12"]

    Returns:
        排序后的 session id 列表，如 [1, 2]
    """
    session_ids: Set[int] = set()

    for ev in evidence:
        # 匹配 "D{数字}:" 格式
        match = re.match(r'D(\d+):', ev)
        if match:
            session_id = int(match.group(1))
            session_ids.add(session_id)

    return sorted(list(session_ids))


def process_locomo_file(file_path: Path) -> None:
    """
    处理单个 locomo*.json 文件，添加 where_ground_truth 字段

    Args:
        file_path: locomo*.json 文件路径
    """
    print(f"Processing {file_path.name}...")

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理每个对话实例
    for item in data:
        if 'qa' not in item:
            continue

        # 处理每个 QA
        for qa in item['qa']:
            evidence = qa.get('evidence', [])

            # 生成 where_ground_truth
            where_ground_truth = extract_session_ids_from_evidence(evidence)

            # 添加到 QA 中
            qa['where_ground_truth'] = where_ground_truth

            # 打印一些示例（前5个）
            if item['qa'].index(qa) < 5:
                print(f"  Q: {qa['question'][:50]}...")
                print(f"    evidence: {evidence}")
                print(f"    where: {qa.get('where')}")
                print(f"    where_ground_truth: {where_ground_truth}")

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Updated {file_path.name}\n")


def main():
    """主函数"""
    # 获取 data/locomo 目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    locomo_dir = project_root / 'data' / 'locomo'

    if not locomo_dir.exists():
        print(f"Error: Directory not found: {locomo_dir}")
        return

    # 查找所有 locomo*.json 文件
    locomo_files = sorted(locomo_dir.glob('locomo*.json'))

    if not locomo_files:
        print(f"No locomo*.json files found in {locomo_dir}")
        return

    print(f"Found {len(locomo_files)} locomo files\n")

    # 处理每个文件
    for file_path in locomo_files:
        process_locomo_file(file_path)

    print(f"[OK] All done! Processed {len(locomo_files)} files.")


if __name__ == '__main__':
    main()
