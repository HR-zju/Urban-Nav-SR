#!/usr/bin/env python3
"""
SR-Nav 数据处理模块使用示例

这个文件展示如何在其他项目中调用 srnav_processor 模块
"""

from pathlib import Path

# ========== 方式 1: 使用快捷函数（最简单） ==========
print("=" * 60)
print("方式 1: 使用快捷函数")
print("=" * 60)

from srnav_processor import (
    create_processor,
    process_mcap,
    process_chunk,
    batch_process_chunks,
)

# 示例 1: 处理单个 MCAP 文件
print("\n--- 示例 1: 处理单个 MCAP ---")
# mcap_path = Path("/media/ubuntu/data1/SR-Nav Dataset/chunk_26/mcap/xxx.mcap")
# success = process_mcap(mcap_path, sample_fps=20.0)
# print(f"处理结果: {success}")

# 示例 2: 处理单个 chunk
print("\n--- 示例 2: 处理单个 chunk ---")
# result = process_chunk("chunk_26", num_workers=4, skip_existing=True)
# print(f"{result.chunk_name}: {result.processed_count}/{result.total_count}")

# 示例 3: 批量处理多个 chunks
print("\n--- 示例 3: 批量处理 ---")
# results = batch_process_chunks(
#     ["chunk_26", "chunk_27", "chunk_28", "chunk_29"],
#     num_workers=4,
#     skip_existing=True
# )
# for r in results:
#     print(f"{r.chunk_name}: {r.processed_count}/{r.total_count}, Success={r.success}")


# ========== 方式 2: 使用类（更灵活） ==========
print("\n" + "=" * 60)
print("方式 2: 使用 SRNavProcessor 类")
print("=" * 60)

from srnav_processor import SRNavProcessor, ProcessingConfig

# 配置
config = ProcessingConfig(
    dataset_root=Path("/media/ubuntu/data1/SR-Nav Dataset"),
    sidewalk_root=Path("/media/ubuntu/data1/SR-Nav Dataset/sidewalk"),
    temp_dir=Path("/tmp/chunk_mcap_temp"),
    sample_fps=20.0,
    num_workers=4,
    generate_bev=False,
    generate_phase2=False,
    # sam3_env_name="sam3",
    # sam3_checkpoint_path=Path("/path/to/checkpoint.pt"),
    # gpus="0",
)

# 创建处理器
processor = SRNavProcessor(config)

# 示例 1: 处理单个 MCAP
print("\n--- 类示例 1: 处理单个 MCAP ---")
# mcap_path = Path("/path/to/file.mcap")
# result = processor.process_single_mcap(mcap_path, skip_existing=True)
# print(f"Success: {result.success}")

# 示例 2: 处理单个 chunk
print("\n--- 类示例 2: 处理单个 chunk ---")
# result = processor.process_chunk("chunk_26", skip_existing=True)
# print(f"Processed: {result.processed_count}/{result.total_count}")

# 示例 3: 检查进度
print("\n--- 类示例 3: 检查进度 ---")
processed, total = processor.get_chunk_progress("chunk_26")
print(f"chunk_26 进度: {processed}/{total} ({processed/total*100:.1f}%)")

# 示例 4: 为单个 ride 生成 BEV
print("\n--- 类示例 4: 生成 BEV ---")
# ride_dir = Path("/path/to/ride/dir")
# success = processor.generate_bev_for_ride(ride_dir)

# 示例 5: 为单个 ride 生成 Phase2
print("\n--- 类示例 5: 生成 Phase2 ---")
# ride_dir = Path("/path/to/ride/dir")
# success = processor.generate_phase2_for_ride(ride_dir)


# ========== 方式 3: 分阶段完整处理 ==========
print("\n" + "=" * 60)
print("方式 3: 分阶段完整处理流程")
print("=" * 60)


def full_pipeline_example():
    """完整处理流程示例"""
    processor = create_processor(num_workers=4)

    # 阶段 1: 基础转换（MCAP → sidewalk 格式）
    print("阶段 1: 基础转换...")
    chunks = ["chunk_26", "chunk_27", "chunk_28", "chunk_29"]
    results = processor.process_chunks(chunks, skip_existing=True)
    for r in results:
        print(f"  {r.chunk_name}: {r.processed_count}/{r.total_count}")

    # 阶段 2: 生成 BEV（可选）
    print("\n阶段 2: 生成 BEV...")
    # 需要时配置：
    # processor.config.generate_bev = True
    # results = processor.process_chunks(chunks, skip_existing=True)

    # 阶段 3: 生成 Phase2（可选）
    print("\n阶段 3: 生成 Phase2...")
    # 需要时配置：
    # processor.config.generate_phase2 = True
    # processor.config.sam3_env_name = "sam3"
    # processor.config.sam3_checkpoint_path = Path("/path/to/checkpoint.pt")
    # results = processor.process_chunks(chunks, skip_existing=True)


# ========== 在其他项目中导入使用 ==========
print("\n" + "=" * 60)
print("在其他项目中导入")
print("=" * 60)

print("""
# 在你的项目中：
import sys
from pathlib import Path

# 添加 srnav_processor 所在路径
srnav_tools_dir = Path("/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer/tools")
sys.path.insert(0, str(srnav_tools_dir))

# 导入并使用
from srnav_processor import process_chunk, batch_process_chunks

# 开始处理
result = process_chunk("chunk_26", num_workers=4)
print(result)
""")


if __name__ == "__main__":
    print("SR-Nav 数据处理模块使用示例")
    print("运行此文件查看各示例说明")
    print("\n提示: 取消注释各示例代码来实际运行")
