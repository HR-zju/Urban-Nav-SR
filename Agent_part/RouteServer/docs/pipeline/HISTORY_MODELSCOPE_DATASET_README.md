---
license: apache-2.0
language:
- zh
- en
tags:
- navigation
- robotics
- trajectory
---

# UrbanNav history 4s / 2Hz dataset

本数据集为 UrbanNav 导航训练提供历史 FPV、可用的 BEV、方向监督和未来轨迹。
仓库地址：`ricksu/SideWalkDataset`。仓库名沿用 SideWalkDataset，实际包含 EgoWalk、SideWalk、Foxglove 三个来源。

## 发布状态与文件布局

本次发布采用混合布局：EgoWalk、SideWalk 保留逐文件目录；Foxglove 使用独立 tar 分片。
Foxglove 每片按完整 clip 分组，目标大小约 8 GiB。tar 不做额外压缩，以降低 JPEG 数据的打包和解包开销。
分片解压后恢复同一训练目录结构，manifest 无需重新生成。

**上传可能仍在进行。** `metadata/foxglove_shards.json` 是计划及本地验证清单，不能单独证明远端上传完成。
确认 `metadata/shard_release_status.json` 存在且标记上传结束，并核对全部分片与 `SHA256SUMS` 后再使用 Foxglove。
远端遗留的 `foxglove/packages/clips/` 为早期未完成的小文件上传；下载时忽略它，以 tar 分片为完整 Foxglove 来源。

| 来源 | Stage II 样本 | 方向样本 | 分发方式 |
| --- | ---: | ---: | --- |
| EgoWalk | 168,583 | 168,583 | `egowalk/packages/clips/` |
| SideWalk | 30,428 | 30,428 | `sidewalk/packages/clips/` |
| Foxglove | 40,042 | 40,042 | `shards/foxglove/*.tar`，解压到根目录 |
| 合计 | 239,053 | 239,053 | 同一观察可同时具有两种训练视图，不将两列相加计数 |

上述是本地生成清单的行数。EgoWalk 逐文件上传曾跳过以下 5 个文件；引用它们的样本需过滤或补齐，不能宣称全部行均可直接训练：

```text
egowalk/packages/clips/000138/media/fpv/004305.jpg
egowalk/packages/clips/000138/media/fpv/004337.jpg
egowalk/packages/clips/000138/media/fpv/004633.jpg
egowalk/packages/clips/000139/media/fpv/001202.jpg
egowalk/packages/clips/000139/media/fpv/001290.jpg
```

完整错误记录见 `metadata/upload_skipped_files.json`。这些是传输失败记录，不是源数据损坏结论。
Foxglove 分片从完整的本地 portable 数据生成。

## 输入输出合同

- 输入历史覆盖 `[t−4s, t]`，2 Hz，偏移为 `[-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0]` 秒。
- **包含当前帧，共 9 个 FPV 时间槽**，合同为 `history_4s_2hz_9f_v1`；不是旧 5 帧或 8 帧合同。
- 每个时间槽选择目标时间之前最近的源帧；不做重复帧 padding。具体滞后容差见 `metadata/history_contract.json`。
- BEV 按 observation anchor 做 latest-causal 时间对齐；这项规则只声明时间选择，BEV 内容及来源限制以生成记录为准。
- 方向合同为 `ego_degree10_12way_terminal_centroid_v2`。
- Stage II 为未来 4 秒的 `[32,3]` metric FLU cumulative pose 轨迹，query 数量 16；没有独立速度监督目标。
- `images` 路径相对于下载并解压后的数据集根目录。`external-source://` 仅用于来源追踪，不是可下载媒体路径。
- 原始传感器同步验证、语义标签质量和训练收敛不由打包/上传检查证明；尤其 Foxglove 以 prepared 数据层面对齐证据为准。

## 下载

安装 ModelScope SDK 后运行以下命令。脚本会忽略早期 Foxglove 零散上传文件，同时下载分片、清单和其他两个来源的数据。

```python
from modelscope.hub.snapshot_download import dataset_snapshot_download

dataset_snapshot_download(
    dataset_id="ricksu/SideWalkDataset",
    local_dir="./SideWalkDataset",
    ignore_patterns=["foxglove/packages/clips/*"],
    max_workers=8,
)
```

仅下载 Foxglove 及相关元数据时，增加：

```python
allow_patterns=["README.md", "SHA256SUMS", "metadata/*", "foxglove/manifests/*",
                "foxglove/packages/clips.jsonl", "foxglove/packages/summary.json",
                "shards/foxglove/*"]
```

重复执行相同下载命令可复用 SDK 缓存。发布完成后，正式实验应记录所用仓库 revision。

## 校验与解压

在数据集根目录执行：

```bash
cd SideWalkDataset
sha256sum -c SHA256SUMS
for shard in shards/foxglove/*.tar; do
    tar -xf "$shard" -C . || exit 1
done
```

tar 成员自带 `foxglove/packages/clips/...` 前缀，请直接解压到根目录，避免产生重复的 `foxglove/foxglove` 层。
分片保留时，Foxglove 下载与解压同时占用约两份空间（约 310 GiB，实际以索引和文件系统为准）。
解压并验证后可自行移走分片；训练只需要解压文件和 manifests。

```text
SideWalkDataset/
  README.md
  SHA256SUMS
  metadata/
    history_contract.json
    generation_receipt.json
    exclusions.json
    foxglove_shards.json
    upload_skipped_files.json
    shard_release_status.json
  egowalk/{manifests,packages}/
  sidewalk/{manifests,packages}/
  foxglove/
    manifests/
    packages/clips.jsonl
    packages/summary.json
    packages/clips/             # 从 tar 解压得到
  shards/foxglove/*.tar
```

## 训练前检查

Stage II 清单位于 `<dataset>/manifests/<dataset>_stage_ii_direct_action_q16.jsonl`；
方向清单位于 `<dataset>/manifests/<dataset>_observations_direction12.jsonl`。
使用绝对数据集根目录解析 `images`，或由训练适配器显式配置该根目录。
以下检查不会修改原始 manifests，会输出缺失媒体对应的行号，方便过滤或补传后复查：

```python
import json
from pathlib import Path

root = Path("./SideWalkDataset").resolve()
for dataset in ("egowalk", "sidewalk", "foxglove"):
    manifest = root / dataset / "manifests" / f"{dataset}_stage_ii_direct_action_q16.jsonl"
    missing_rows = 0
    with manifest.open() as stream:
        for number, line in enumerate(stream, 1):
            sample = json.loads(line)
            missing = [path for path in sample["images"] if not (root / path).is_file()]
            if missing:
                missing_rows += 1
                print(dataset, number, missing)
    print(dataset, "missing_media_rows=", missing_rows)
```

生成时跳过的损坏或不满足合同的数据见 `metadata/exclusions.json`；上传缺失与生成排除是两个不同阶段。

## 许可与来源

仓库页面配置为 Apache-2.0。各来源数据及媒体仍需遵守其原始许可和使用条件；本 README 不构成对上游媒体的重新授权。
来源及生成过程见 `metadata/generation_receipt.json` 和 package manifests。
