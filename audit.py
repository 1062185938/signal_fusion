#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对照 review_*.json 与 audit.json，按 analysis_id 匹配，
输出包含 true_label / recommended_label / iq_label 的 CSV。
"""

import json
import csv
import glob
import os
from pathlib import Path

# ============ 配置路径 ============
REVIEW_DIR = "/home/dianci/projects/signal_fusion/outputs/technology_recognition_v2_location_validation/cross_location_llm_adjudication_v0_responses"
AUDIT_FILE = "/home/dianci/projects/signal_fusion/outputs/technology_recognition_v2_location_validation/private/cross_location_llm_adjudication_v1_audit.json"
OUTPUT_CSV = "/home/dianci/projects/signal_fusion/outputs/technology_recognition_v2_location_validation/review_vs_audit_stats.csv"


def load_reviews(review_dir: str) -> dict:
    """读取目录下所有 review_*.json，返回 {analysis_id: data}"""
    reviews = {}
    pattern = os.path.join(review_dir, "review_*.json")
    files = sorted(glob.glob(pattern))
    print(f"[INFO] 在 {review_dir} 找到 {len(files)} 个 review 文件")

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] 读取失败 {fp}: {e}")
            continue

        aid = data.get("analysis_id")
        if not aid:
            print(f"[WARN] 文件 {fp} 缺少 analysis_id，已跳过")
            continue
        reviews[aid] = data

    print(f"[INFO] 有效 review 条目: {len(reviews)}")
    return reviews


def load_audit(audit_file: str) -> dict:
    """读取 audit.json，建立 {analysis_id: [case, ...]} 索引（A、B 两侧都索引）"""
    with open(audit_file, "r", encoding="utf-8") as f:
        audit = json.load(f)

    cases = audit.get("cases", [])
    print(f"[INFO] audit 文件中共有 {len(cases)} 个 case")

    index = {}
    for case in cases:
        for key in ("set_a_analysis_id", "set_b_analysis_id"):
            aid = case.get(key)
            if aid:
                index.setdefault(aid, []).append(case)

    print(f"[INFO] 索引到 {len(index)} 个不同的 analysis_id")
    return index


def main():
    reviews = load_reviews(REVIEW_DIR)
    audit_index = load_audit(AUDIT_FILE)

    rows = []
    missing_in_audit = 0          # review 里有，audit 里找不到
    matched_count = 0             # 成功匹配的 (review, case) 对数

    for aid, review in reviews.items():
        rec_label = review.get("recommended_label", "")
        decision = review.get("decision", "")
        conf = review.get("confidence_level", "")

        matched_cases = audit_index.get(aid, [])
        if not matched_cases:
            missing_in_audit += 1
            # 即使没有匹配，也输出一行方便核对
            rows.append({
                "analysis_id": aid,
                "decision": decision,
                "confidence_level": conf,
                "true_label": "",
                "recommended_label": rec_label,
                "iq_label": "",
                "feature_label": "",
                "equal_fusion_label": "",
                "fold": "",
                "location": "",
            })
            continue

        for case in matched_cases:
            matched_count += 1
            rows.append({
                "analysis_id": aid,
                "decision": decision,
                "confidence_level": conf,
                "true_label": case.get("true_label", ""),
                "recommended_label": rec_label,
                "iq_label": case.get("iq_label", ""),
                "feature_label": case.get("feature_label", ""),
                "equal_fusion_label": case.get("equal_fusion_label", ""),
                "fold": case.get("fold", ""),
                "location": case.get("location", ""),
            })

    # 写出 CSV
    fieldnames = [
        "analysis_id", "decision", "confidence_level",
        "true_label", "recommended_label", "iq_label",
        "feature_label", "equal_fusion_label", "fold", "location",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ============ 统计摘要 ============
    print("\n========== 统计摘要 ==========")
    print(f"review 文件总数        : {len(reviews)}")
    print(f"成功匹配的 (review,case): {matched_count}")
    print(f"在 audit 中找不到的 review: {missing_in_audit}")
    print(f"CSV 行数               : {len(rows)}")
    print(f"CSV 已保存到           : {OUTPUT_CSV}")

    # 简单的一致性统计（只看成功匹配的）
    matched_rows = [r for r in rows if r["true_label"]]
    if matched_rows:
        agree = sum(1 for r in matched_rows if r["true_label"] == r["recommended_label"])
        print(f"\nrecommended_label 与 true_label 一致: {agree}/{len(matched_rows)}"
              f" ({agree / len(matched_rows) * 100:.2f}%)")

        # 按 true_label 分组统计
        from collections import Counter
        print("\n-- true_label 分布 --")
        for k, v in Counter(r["true_label"] for r in matched_rows).most_common():
            print(f"  {k}: {v}")

        print("\n-- recommended_label 分布 --")
        for k, v in Counter(r["recommended_label"] for r in matched_rows).most_common():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()