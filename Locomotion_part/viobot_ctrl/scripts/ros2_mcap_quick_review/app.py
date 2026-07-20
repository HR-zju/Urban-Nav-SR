from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from review_tool.annotations import build_manifest, load_annotations, save_annotation
from review_tool.config import load_config
from review_tool.dataset import load_dataset
from review_tool.io_utils import read_json

st.set_page_config(page_title="ROS2 / MCAP 快速审核", layout="wide")

@st.cache_data(ttl=2)
def cached_dataset(path: str) -> pd.DataFrame:
    return load_dataset(path)

@st.cache_data(ttl=2)
def cached_annotations(path: str) -> pd.DataFrame:
    return load_annotations(path)

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

    annotation_map = {row["bag_id"]: row.to_dict() for _, row in annotations.iterrows()}
    dataset = dataset.copy()
    dataset["label"] = dataset["bag_id"].map(
        lambda bag_id: annotation_map.get(bag_id, {}).get("label", "")
    )

    st.sidebar.title("审核队列")
    filter_mode = st.sidebar.radio(
        "显示", ["未审核", "全部", "Good", "Bad", "预处理错误"], index=0
    )
    if filter_mode == "未审核":
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
    dl1, dl2 = st.columns(2)
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

if __name__ == "__main__":
    main()
