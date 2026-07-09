#!/usr/bin/env python3
"""
统计 JSON 文件中每个场景（scene_id）的样本数量
从 video 字段提取场景编号，格式: images/<scene_id>_<type>_<number>
用法: python count_scenes.py <json_file> [--csv OUTPUT_CSV] [--by-type]
基本用法（只统计总数）
bashpython count_scenes.py your_data.json
按任务类型细分（r2r / cvdn / dun / oln 等）
bashpython count_scenes.py your_data.json --by-type
同时导出 CSV
bashpython count_scenes.py your_data.json --by-type --csv output.csv
"""

import json
import re
import sys
import argparse
from collections import defaultdict


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass
    records = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [警告] 第 {lineno} 行解析失败，已跳过: {e}")
    return records


def extract_scene_and_type(video_path):
    """
    从 video 字段提取 scene_id 和 task_type
    例: images/17DRP5sb8fy_r2r_001803  ->  ('17DRP5sb8fy', 'r2r')
        images/82sE5b5pLXE_cvdn_000000 ->  ('82sE5b5pLXE', 'cvdn')
    """
    basename = video_path.split("/")[-1]           # 去掉目录前缀
    m = re.match(r"^([A-Za-z0-9]+)_([^_]+)_\d+", basename)
    if m:
        return m.group(1), m.group(2).lower()
    # 兜底：取第一个下划线前的内容作为 scene_id
    parts = basename.split("_")
    return parts[0], "unknown"


def compute_counts(records, by_type=False):
    scene_total = defaultdict(int)       # scene_id -> 总数
    scene_by_type = defaultdict(lambda: defaultdict(int))  # scene_id -> type -> 数

    skipped = 0
    for rec in records:
        video = rec.get("video") or rec.get("Video")
        if not video:
            skipped += 1
            continue
        scene_id, task_type = extract_scene_and_type(video)
        scene_total[scene_id] += 1
        scene_by_type[scene_id][task_type] += 1

    return scene_total, scene_by_type, skipped


def print_report(scene_total, scene_by_type, skipped, total, by_type):
    all_types = sorted({t for types in scene_by_type.values() for t in types})

    print("\n" + "=" * 72)
    print(f"  场景样本数量统计报告  (共 {total} 条记录，跳过 {skipped} 条无效记录)")
    print("=" * 72)

    if by_type and all_types:
        type_cols = "".join(f"  {t:>8}" for t in all_types)
        print(f"  {'scene_id':<22}  {'总计':>6}{type_cols}")
        print("-" * 72)
        for scene_id in sorted(scene_total, key=lambda x: scene_total[x], reverse=True):
            type_vals = "".join(f"  {scene_by_type[scene_id].get(t, 0):>8}" for t in all_types)
            print(f"  {scene_id:<22}  {scene_total[scene_id]:>6}{type_vals}")
    else:
        print(f"  {'scene_id':<22}  {'样本数':>8}")
        print("-" * 72)
        for scene_id in sorted(scene_total, key=lambda x: scene_total[x], reverse=True):
            print(f"  {scene_id:<22}  {scene_total[scene_id]:>8}")

    print("-" * 72)
    print(f"  共 {len(scene_total)} 个场景，{sum(scene_total.values())} 个有效样本")
    print("=" * 72 + "\n")


def save_csv(scene_total, scene_by_type, output_path, by_type):
    import csv
    all_types = sorted({t for types in scene_by_type.values() for t in types})
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if by_type and all_types:
            writer.writerow(["scene_id", "total"] + all_types)
            for scene_id in sorted(scene_total, key=lambda x: scene_total[x], reverse=True):
                row = [scene_id, scene_total[scene_id]] + [scene_by_type[scene_id].get(t, 0) for t in all_types]
                writer.writerow(row)
        else:
            writer.writerow(["scene_id", "count"])
            for scene_id in sorted(scene_total, key=lambda x: scene_total[x], reverse=True):
                writer.writerow([scene_id, scene_total[scene_id]])
    print(f"  CSV 结果已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="统计 JSON 文件中每个场景的样本数量")
    parser.add_argument("json_file", help="输入 JSON / JSONL 文件路径")
    parser.add_argument("--csv", metavar="OUTPUT_CSV", help="可选：将结果导出为 CSV 文件")
    parser.add_argument("--by-type", action="store_true",
                        help="按任务类型（r2r / cvdn / dun 等）分别统计")
    args = parser.parse_args()

    print(f"\n正在读取文件: {args.json_file}")
    records = load_json(args.json_file)
    print(f"共加载 {len(records)} 条记录")

    scene_total, scene_by_type, skipped = compute_counts(records, args.by_type)
    print_report(scene_total, scene_by_type, skipped, len(records), args.by_type)

    if args.csv:
        save_csv(scene_total, scene_by_type, args.csv, args.by_type)


if __name__ == "__main__":
    main()