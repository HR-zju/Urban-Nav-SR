from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from review_tool.annotations import build_manifest, load_annotations, save_annotation
from review_tool.config import load_config
from review_tool.dataset import load_dataset
from review_tool.initial_decisions import (
    MATCHED,
    attach_initial_decisions,
    is_selected_manual_decision,
    load_initial_decisions,
)
from review_tool.io_utils import read_json

st.set_page_config(page_title="ROS2 / MCAP 快速审核", layout="wide")

@st.cache_data(ttl=2)
def cached_dataset(path: str) -> pd.DataFrame:
    return load_dataset(path)

@st.cache_data(ttl=2)
def cached_annotations(path: str) -> pd.DataFrame:
    return load_annotations(path)

@st.cache_data(ttl=2)
def cached_initial_decisions(path: str) -> pd.DataFrame:
    return load_initial_decisions(path)

def invalidate_cache():
    cached_dataset.clear()
    cached_annotations.clear()

def move_to(rows: pd.DataFrame, current_id: str, step: int):
    ids = rows["bag_id"].tolist()
    if not ids:
        return
    try:
        index = ids.index(current_id)
    except ValueError:
        index = 0
    st.session_state.current_bag_id = ids[max(0, min(len(ids) - 1, index + step))]

def main():
    config_path = os.environ.get("REVIEW_CONFIG", "config.yaml")
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        st.error(f"配置加载失败：{exc}")
        st.code(
            "cp config.example.yaml config.yaml\n"
            "export REVIEW_CONFIG=$PWD/config.yaml\n"
            "streamlit run app.py"
        )
        st.stop()

    work_dir = Path(cfg["paths"]["work_dir"])
    dataset_path = work_dir / "dataset.csv"
    annotations_path = work_dir / "annotations.csv"
    dataset = cached_dataset(str(dataset_path))
    annotations = cached_annotations(str(annotations_path))
    if dataset.empty:
        st.warning("数据清单为空，请先运行 scan_files.py。")
        st.stop()

    initial_cfg = cfg.get("initial_decisions", {})
    initial_enabled = False
    initial_error = ""
    initial_decisions = pd.DataFrame()
    selected_manual_decisions = initial_cfg.get(
        "selected_manual_decisions", ["KEEP", "REVIEW"]
    )
    if bool(initial_cfg.get("enabled", False)):
        try:
            initial_csv_path = str(initial_cfg.get("csv_path", "")).strip()
            if not initial_csv_path:
                raise ValueError("initial_decisions.csv_path 未配置")
            initial_decisions = cached_initial_decisions(initial_csv_path)
            dataset = attach_initial_decisions(dataset, initial_decisions)
            initial_enabled = True
        except Exception as exc:
            initial_error = f"初筛决定加载失败：{type(exc).__name__}: {exc}"

    annotation_map = {row["bag_id"]: row.to_dict() for _, row in annotations.iterrows()}
    dataset = dataset.copy()
    dataset["label"] = dataset["bag_id"].map(
        lambda bag_id: annotation_map.get(bag_id, {}).get("label", "")
    )

    st.sidebar.title("审核队列")
    if initial_error:
        st.sidebar.warning(initial_error)
    selected_mask = pd.Series(False, index=dataset.index)
    filter_options = ["未审核", "全部", "Good", "Bad", "预处理错误"]
    if initial_enabled:
        selected_mask = is_selected_manual_decision(
            dataset, selected_manual_decisions
        )
        filter_options = [
            "初筛人工候选—未二筛",
            "初筛人工候选—全部",
            *filter_options,
            "初筛匹配异常",
        ]
    filter_mode = st.sidebar.radio(
        "显示", filter_options, index=0
    )
    if filter_mode == "初筛人工候选—未二筛":
        rows = dataset[
            selected_mask
            & (dataset["label"] == "")
            & dataset["status"].isin(["ready", "error"])
        ].copy()
    elif filter_mode == "初筛人工候选—全部":
        rows = dataset[
            selected_mask & dataset["status"].isin(["ready", "error"])
        ].copy()
    elif filter_mode == "初筛匹配异常":
        rows = dataset[
            dataset["initial_match_status"].ne(MATCHED)
            & dataset["status"].isin(["ready", "error"])
        ].copy()
    elif filter_mode == "未审核":
        rows = dataset[
            (dataset["label"] == "") & dataset["status"].isin(["ready", "error"])
        ].copy()
    elif filter_mode == "Good":
        rows = dataset[dataset["label"] == "Good"].copy()
    elif filter_mode == "Bad":
        rows = dataset[dataset["label"] == "Bad"].copy()
    elif filter_mode == "预处理错误":
        rows = dataset[dataset["status"] == "error"].copy()
    else:
        rows = dataset[dataset["status"].isin(["ready", "error"])].copy()

    total = len(dataset)
    good_count = int((dataset["label"] == "Good").sum())
    bad_count = int((dataset["label"] == "Bad").sum())
    reviewed = good_count + bad_count
    st.sidebar.metric("总数", total)
    st.sidebar.metric("已审核", reviewed)
    st.sidebar.metric("Good / Bad", f"{good_count} / {bad_count}")
    st.sidebar.progress(reviewed / total if total else 0.0)
    if initial_enabled:
        matched_count = int(dataset["initial_match_status"].eq(MATCHED).sum())
        selected_count = int(selected_mask.sum())
        selected_good = int((selected_mask & dataset["label"].eq("Good")).sum())
        selected_bad = int((selected_mask & dataset["label"].eq("Bad")).sum())
        selected_unreviewed = int((selected_mask & dataset["label"].eq("")).sum())
        with st.sidebar.expander("初筛关联统计", expanded=True):
            st.write(f"CSV 记录：{len(initial_decisions)}")
            st.write(f"成功匹配：{matched_count} / {total}")
            selected_text = " / ".join(selected_manual_decisions)
            st.write(f"人工候选（{selected_text}）：{selected_count}")
            st.write(f"二筛未审核：{selected_unreviewed}")
            st.write(f"二筛 Good / Bad：{selected_good} / {selected_bad}")
            problems = dataset[dataset["initial_match_status"].ne(MATCHED)][
                ["bag_id", "bag_path", "initial_record_id", "initial_match_status"]
            ]
            st.download_button(
                "下载匹配异常 CSV",
                data=problems.to_csv(index=False).encode("utf-8-sig"),
                file_name="initial_decision_match_problems.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if rows.empty:
        st.success("当前筛选条件下没有数据。")
        if not annotations.empty:
            st.download_button(
                "下载 Good 清单",
                data=build_manifest(annotations, "Good"),
                file_name="good_files.txt",
                mime="text/plain",
            )
        st.stop()

    if "current_bag_id" not in st.session_state:
        st.session_state.current_bag_id = rows.iloc[0]["bag_id"]
    if st.session_state.current_bag_id not in set(rows["bag_id"]):
        st.session_state.current_bag_id = rows.iloc[0]["bag_id"]

    ids = rows["bag_id"].tolist()
    current_index = ids.index(st.session_state.current_bag_id)
    labels = {
        bag_id: f"{idx + 1}/{len(ids)}  {Path(rows.loc[rows['bag_id'] == bag_id, 'bag_path'].iloc[0]).name}"
        for idx, bag_id in enumerate(ids)
    }
    selected_id = st.sidebar.selectbox(
        "跳转", options=ids, index=current_index, format_func=lambda bag_id: labels[bag_id]
    )
    if selected_id != st.session_state.current_bag_id:
        st.session_state.current_bag_id = selected_id
        st.rerun()

    row = rows[rows["bag_id"] == st.session_state.current_bag_id].iloc[0]
    bag_id, bag_path = row["bag_id"], row["bag_path"]
    cache_dir = Path(row["cache_dir"])
    video_path, info_path = cache_dir / "review.mp4", cache_dir / "info.json"
    existing = annotation_map.get(bag_id, {})

    st.title("ROS2 / MCAP 数据快速审核")
    st.caption(f"{current_index + 1} / {len(rows)} ｜ {bag_path}")
    if initial_enabled:
        status = row["initial_match_status"]
        if status == MATCHED:
            issues = row["initial_issues"] or "-"
            st.info(
                f"初筛人工决定：{row['initial_manual_decision'] or '-'} ｜ "
                f"自动建议：{row['initial_auto_recommendation'] or '-'} ｜ "
                f"记录 ID：{row['initial_record_id']} ｜ 问题：{issues}"
            )
            st.caption(f"初筛来源文件：{row['initial_source_name']}")
        else:
            status_labels = {
                "no_record_id": "文件名中没有 rec_ 记录 ID",
                "not_in_decisions": "记录 ID 不在初筛 CSV 中",
                "decision_id_ambiguous": "初筛 CSV 中记录 ID 重复",
                "dataset_id_ambiguous": "审核数据中记录 ID 重复或目录内有多个 ID",
            }
            st.warning(
                "初筛匹配状态："
                + status_labels.get(status, status or "未知")
            )

    left, mid, _ = st.columns([1, 1, 5])
    if left.button("← 上一条", use_container_width=True):
        move_to(rows, bag_id, -1)
        st.rerun()
    if mid.button("下一条 →", use_container_width=True):
        move_to(rows, bag_id, 1)
        st.rerun()

    if row["status"] == "error":
        st.error(f"预处理失败：{row['error']}")
    if video_path.exists():
        st.video(str(video_path), format="video/mp4")
    else:
        st.warning("review.mp4 不存在，可标记 Bad 或重新预处理。")

    info = read_json(info_path, {}) or {}
    if info:
        warnings = info.get("warnings", [])
        if warnings:
            st.warning("；".join(str(x) for x in warnings))
        with st.expander("解析信息"):
            st.json(info)

    reasons = cfg["review"]["bad_reasons"]
    old_reason = existing.get("bad_reason", "")
    reason = st.selectbox(
        "Bad 原因",
        reasons,
        index=reasons.index(old_reason) if old_reason in reasons else 0,
        key=f"reason_{bag_id}",
    )
    comment = st.text_input(
        "备注（可选）",
        value=existing.get("comment", ""),
        key=f"comment_{bag_id}",
    )

    good_col, bad_col, skip_col = st.columns([1, 1, 3])
    if good_col.button("✅ Good", type="primary", use_container_width=True):
        save_annotation(annotations_path, bag_id, bag_path, "Good", "", comment)
        invalidate_cache()
        move_to(rows, bag_id, 1)
        st.rerun()
    if bad_col.button("❌ Bad", use_container_width=True):
        save_annotation(annotations_path, bag_id, bag_path, "Bad", reason, comment)
        invalidate_cache()
        move_to(rows, bag_id, 1)
        st.rerun()
    if skip_col.button("跳过并进入下一条"):
        move_to(rows, bag_id, 1)
        st.rerun()

    if existing:
        st.info(
            f"当前标签：{existing.get('label', '')} ｜ "
            f"原因：{existing.get('bad_reason', '') or '-'} ｜ "
            f"时间：{existing.get('reviewed_at', '')}"
        )

    st.divider()
    download_columns = 3 if initial_enabled else 2
    download_cols = st.columns(download_columns)
    dl1, dl2 = download_cols[:2]
    dl1.download_button(
        "下载 Good 文件清单",
        data=build_manifest(annotations, "Good"),
        file_name="good_files.txt",
        mime="text/plain",
        use_container_width=True,
    )
    dl2.download_button(
        "下载完整标注 CSV",
        data=annotations.to_csv(index=False).encode("utf-8-sig"),
        file_name="annotations.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if initial_enabled:
        selected_ids = set(dataset.loc[selected_mask, "bag_id"])
        selected_annotations = annotations[
            annotations["bag_id"].isin(selected_ids)
        ]
        download_cols[2].download_button(
            "下载初筛候选中的二筛 Good",
            data=build_manifest(selected_annotations, "Good"),
            file_name="initial_candidates_second_review_good.txt",
            mime="text/plain",
            use_container_width=True,
        )

if __name__ == "__main__":
    main()
