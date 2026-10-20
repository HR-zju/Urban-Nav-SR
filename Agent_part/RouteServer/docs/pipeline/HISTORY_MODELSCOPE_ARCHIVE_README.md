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

# UrbanNav history 4s / 2Hz — 全量分片数据集

`ricksu/SideWalkDataset` 包含 EgoWalk、SideWalk 和 Foxglove 三个来源的导航训练数据。
媒体、每个 clip 的标签和 review，以及汇总训练清单均通过 tar 包分发。
每片媒体按完整 clip 分组，目标大小约 8 GiB；JPEG 已压缩，tar 使用无额外压缩模式。
解压后恢复原目录布局，训练无需读取 tar 或更改数据格式。

## 发布状态

本布局以 `metadata/archive_release_status.json` 为完成标记，状态应为 `complete`。
下载清单为 `metadata/all_archives.json`，所有包的校验值见 `SHA256SUMS`。
迁移期间旧零散目录可能暂时可见；下载命令明确只获取分片和发布元数据。
全部分片远端大小及 SHA-256 核对通过后，旧 `egowalk/`、`sidewalk/`、`foxglove/` 目录从当前分支移除。
在本地解压时这些目录会重新生成。

旧混合发布的 `metadata/shard_release_status.json` 和 `metadata/upload_skipped_files.json`
仅记录历史过程。旧版 EgoWalk 上传跳过的 5 张图片已纳入本次完整源数据归档；本次是否完成以新的发布标记为准。

| 来源 | Stage II 行数 | 方向监督行数 | 媒体分片 |
| --- | ---: | ---: | --- |
| EgoWalk | 168,583 | 168,583 | `shards/egowalk/*.tar` |
| SideWalk | 30,428 | 30,428 | `shards/sidewalk/*.tar` |
| Foxglove | 40,042 | 40,042 | `shards/foxglove/*.tar` |
| 合计 | 239,053 | 239,053 | 两列是相同 observation 的不同视图，不相加 |

清单和 package 索引统一放在 `shards/metadata/metadata-00000.tar`。
该包包含三个来源的 `manifests/`、`packages/clips.jsonl`、`packages/summary.json` 和原始生成 metadata。
生成阶段的排除项见解压后的 `metadata/exclusions.json`。

## 数据合同

- observation profile：`history_4s_2hz_9f_v1`。
- 历史范围 `[t−4s, t]`，2 Hz，**包含当前帧共 9 个 FPV 时间槽**。
- 目标偏移：`[-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0]` 秒。
- 取目标时间之前最近的源帧；禁止重复帧 padding。滞后容差见 `metadata/history_contract.json`。
- BEV 按 observation anchor 使用 latest-causal 时间对齐；内容来源限制仍以生成记录为准。
- 方向合同：`ego_degree10_12way_terminal_centroid_v2`。
- Stage II：未来 4 秒 `[32,3]` metric FLU cumulative pose 轨迹，16 个 query，没有独立速度监督目标。
- `images` 均相对于解压后的数据集根目录；`external-source://` 仅记录来源，不是媒体下载地址。
- prepared 数据对齐、原始传感器同步、语义标注质量和训练收敛是不同验证层；打包校验只证明分发文件一致。

## 下载

安装 ModelScope SDK，然后执行：

```python
from modelscope.hub.snapshot_download import dataset_snapshot_download

dataset_snapshot_download(
    dataset_id="ricksu/SideWalkDataset",
    local_dir="./SideWalkDataset",
    allow_patterns=["README.md", "SHA256SUMS", "metadata/*", "shards/*"],
    max_workers=4,
)
```

此命令过滤旧版零散文件。重复执行可利用 SDK 缓存；正式实验请记录下载的仓库 revision。
只需一个来源时，将 `"shards/*"` 换成 `"shards/metadata/*"` 和例如 `"shards/foxglove/*"`，
并只校验/使用所下载来源。汇总清单包包含三个来源的 manifests，不代表其他来源媒体也已下载。

## 校验和解压

完整下载后执行：

```bash
cd SideWalkDataset
sha256sum -c SHA256SUMS || exit 1
tar -xf shards/metadata/metadata-00000.tar -C . || exit 1
for dataset in egowalk sidewalk foxglove; do
    for shard in "shards/$dataset/"*.tar; do
        tar -xf "$shard" -C . || exit 1
    done
done
```

tar 自带相对根目录路径，所有分片都直接解压到 `SideWalkDataset/`，不要额外套一层来源目录。
完整数据逻辑大小约 300 GiB，实际以分片索引为准；同时保留所有 tar 和解压文件需约 600 GiB，另预留文件系统开销。
可逐片下载、校验、解压后移走 tar 以降低峰值占用。本地校验完成前保留对应 tar。

### 理想平台挂载版

理想平台已挂载完成版只读归档：

```text
/mnt/datasets/ricksu__SideWalkDataset/26-09-25-0429
```

该目录包含 40 个 tar 和发布标记 `metadata/archive_release_status.json`，但不包含解压后的媒体树。
将其解压到共享 GPFS 的独立可写目录：

```text
/lpai/volumes/mind-eb-bd-ga/suh/dataset/urbannav_history_4s_2hz_v1
```

不要解压到旧版 `26-09-23-2317` 的碎文件目录，也不要写入 FUSE 挂载。解压作业须先在挂载根目录
执行 `sha256sum -c SHA256SUMS`，在 staging 目录完成解压和 manifest 存在性检查后，再将 staging
目录原子改名为上述正式路径。这样训练入口不会读取半成品。

解压后的布局：

```text
SideWalkDataset/
  README.md
  SHA256SUMS
  metadata/
    all_archives.json
    archive_release_status.json
    history_contract.json
    generation_receipt.json
    exclusions.json
  egowalk/
    manifests/
    packages/clips.jsonl
    packages/summary.json
    packages/clips/000000/...
  sidewalk/{manifests,packages}/...
  foxglove/{manifests,packages}/...
  shards/{metadata,egowalk,sidewalk,foxglove}/*.tar
```

## 训练入口

Stage II：`<dataset>/manifests/<dataset>_stage_ii_direct_action_q16.jsonl`。
方向监督：`<dataset>/manifests/<dataset>_observations_direction12.jsonl`。
`<dataset>` 为 `egowalk`、`sidewalk` 或 `foxglove`。训练适配器需用解压根目录解析相对图片路径。

下载后可运行媒体存在性检查：

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
            missing = [p for p in sample["images"] if not (root / p).is_file()]
            if missing:
                missing_rows += 1
                print(dataset, number, missing)
    print(dataset, "missing_media_rows=", missing_rows)
```

使用新合同配置模型输入，再完成真实数据的短训练验证。生成清单行数不等于语义质量验收结果。

## 许可与来源

仓库页面配置为 Apache-2.0。各来源数据及媒体仍须遵守其原始许可和使用条件；本 README 不构成对上游媒体的重新授权。
来源见 `metadata/generation_receipt.json` 及各 package manifests。
