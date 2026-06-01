#!/usr/bin/env python3
"""
SR-Nav 数据处理封装模块
提供简洁的 API 接口供其他项目调用

主要功能：
- MCAP → sidewalk 格式转换
- 批量处理 chunks
- BEV 地图生成
- Phase2/SAM 标注生成
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from dataclasses import dataclass, field


@dataclass
class ProcessingConfig:
    """处理配置"""
    dataset_root: Path = Path("/media/ubuntu/data1/SR-Nav Dataset")
    sidewalk_root: Path = Path("/media/ubuntu/data1/SR-Nav Dataset/sidewalk")
    temp_dir: Path = Path("/tmp/chunk_mcap_temp")
    sample_fps: float = 20.0
    num_workers: int = 4
    generate_bev: bool = False
    generate_phase2: bool = False
    generate_walkgpt: bool = False
    sam3_env_name: str = "sam3"
    sam3_checkpoint_path: Optional[Path] = None
    gpus: str = "0"


@dataclass
class ProcessResult:
    """处理结果"""
    success: bool
    chunk_name: str
    processed_count: int
    total_count: int
    output_dir: Path
    error_message: Optional[str] = None


class SRNavProcessor:
    """SR-Nav 数据处理器主类"""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self._dataset_dir = Path(__file__).resolve().parent
        self._tools_dir = self._dataset_dir.parent
        self._model_dir = self._tools_dir / "model_processing"

    def process_single_mcap(
        self,
        mcap_path: Path,
        output_dir: Optional[Path] = None,
        skip_existing: bool = True,
    ) -> ProcessResult:
        """
        处理单个 MCAP 文件

        Args:
            mcap_path: MCAP 文件路径
            output_dir: 输出目录（默认自动推断）
            skip_existing: 跳过已处理的文件

        Returns:
            ProcessResult: 处理结果
        """
        if output_dir is None:
            # 从路径中推断 chunk 和文件名
            parts = mcap_path.parts
            chunk_name = None
            for i, p in enumerate(parts):
                if p.startswith("chunk_"):
                    chunk_name = p
                    break
            if chunk_name is None:
                chunk_name = "chunk_unknown"
            output_dir = self.config.sidewalk_root / chunk_name / mcap_path.stem

        if skip_existing and (output_dir / "frame_gps.npy").exists():
            return ProcessResult(
                success=True,
                chunk_name=chunk_name,
                processed_count=1,
                total_count=1,
                output_dir=output_dir,
            )

        cmd = [
            sys.executable,
            str(self._dataset_dir / "mcap_to_sidewalk_inputs.py"),
            "--mcap-path", str(mcap_path),
            "--output-dir", str(output_dir),
            "--sample-fps", str(self.config.sample_fps),
            "--image-topic", "auto",
            "--gps-topic", "auto",
            "--heading-topic", "auto",
            "--imu-topic", "auto",
            "--pose-topic", "auto",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return ProcessResult(
                success=True,
                chunk_name=chunk_name or "unknown",
                processed_count=1,
                total_count=1,
                output_dir=output_dir,
            )
        except subprocess.CalledProcessError as e:
            return ProcessResult(
                success=False,
                chunk_name=chunk_name or "unknown",
                processed_count=0,
                total_count=1,
                output_dir=output_dir,
                error_message=f"stderr: {e.stderr}\nstdout: {e.stdout}"
            )

    def process_chunk(
        self,
        chunk_name: str,
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ProcessResult:
        """
        处理单个 chunk

        Args:
            chunk_name: chunk 名称，如 "chunk_26"
            skip_existing: 跳过已处理的文件
            progress_callback: 进度回调函数 (processed, total)

        Returns:
            ProcessResult: 处理结果
        """
        chunk_dir = self.config.dataset_root / chunk_name
        mcap_dir = chunk_dir / "mcap"

        if not mcap_dir.exists():
            return ProcessResult(
                success=False,
                chunk_name=chunk_name,
                processed_count=0,
                total_count=0,
                output_dir=self.config.sidewalk_root / chunk_name,
                error_message=f"MCAP dir not found: {mcap_dir}"
            )

        mcap_files = sorted(mcap_dir.glob("*.mcap"))
        total = len(mcap_files)

        # 使用 process_chunks_26_29.py 脚本
        cmd = [
            sys.executable,
            str(self._dataset_dir / "process_chunks_26_29.py"),
            "--dataset-root", str(self.config.dataset_root),
            "--sidewalk-root", str(self.config.sidewalk_root),
            "--temp-dir", str(self.config.temp_dir),
            "--chunks", chunk_name.replace("chunk_", ""),
            "--sample-fps", str(self.config.sample_fps),
            "--num-workers", str(self.config.num_workers),
        ]

        if skip_existing:
            cmd.append("--skip-existing")
        if self.config.generate_bev:
            cmd.append("--generate-bev")
        if self.config.generate_phase2:
            cmd.append("--generate-phase2")
        if self.config.generate_walkgpt:
            cmd.append("--generate-walkgpt")
        if self.config.sam3_checkpoint_path:
            cmd.extend(["--checkpoint-path", str(self.config.sam3_checkpoint_path)])

        try:
            subprocess.run(cmd, check=True, capture_output=False)
            # 统计处理数量
            output_chunk_dir = self.config.sidewalk_root / chunk_name
            processed = len(list(output_chunk_dir.glob("*"))) - 1 if output_chunk_dir.exists() else 0
            return ProcessResult(
                success=True,
                chunk_name=chunk_name,
                processed_count=processed,
                total_count=total,
                output_dir=output_chunk_dir,
            )
        except subprocess.CalledProcessError as e:
            return ProcessResult(
                success=False,
                chunk_name=chunk_name,
                processed_count=0,
                total_count=total,
                output_dir=self.config.sidewalk_root / chunk_name,
                error_message=str(e)
            )

    def process_chunks(
        self,
        chunk_names: Sequence[str],
        skip_existing: bool = True,
    ) -> list[ProcessResult]:
        """
        批量处理多个 chunks

        Args:
            chunk_names: chunk 名称列表，如 ["chunk_26", "chunk_27"]
            skip_existing: 跳过已处理的文件

        Returns:
            list[ProcessResult]: 每个 chunk 的处理结果
        """
        results = []
        for chunk_name in chunk_names:
            result = self.process_chunk(chunk_name, skip_existing)
            results.append(result)
        return results

    def generate_bev_for_ride(
        self,
        ride_dir: Path,
        output_dir: Optional[Path] = None,
    ) -> bool:
        """
        为单个 ride 生成 BEV 地图

        Args:
            ride_dir: sidewalk 格式的 ride 目录
            output_dir: 输出目录（默认同 ride_dir）

        Returns:
            bool: 是否成功
        """
        output_dir = output_dir or ride_dir
        cmd = [
            sys.executable,
            str(self._dataset_dir / "process_sidewalk_dataset.py"),
            "--sidewalk-root", str(ride_dir),
            "--output-root", str(output_dir),
            "--sample-fps", str(self.config.sample_fps),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def generate_phase2_for_ride(
        self,
        ride_dir: Path,
        generate_video: bool = True,
        cleanup_temp: bool = True,
    ) -> bool:
        """
        为单个 ride 生成 Phase2 标注

        Args:
            ride_dir: sidewalk 格式的 ride 目录
            generate_video: 是否生成调试视频
            cleanup_temp: 是否清理临时文件

        Returns:
            bool: 是否成功
        """
        cmd = [
            sys.executable,
            str(self._model_dir / "run_sam3_som_phase.py"),
            "--clip-dir", str(ride_dir),
            "--run-sam3",
            "--sam3-env-name", self.config.sam3_env_name,
            "--gpus", self.config.gpus,
            "--sample-fps", str(self.config.sample_fps),
            "--disable-route-support",
        ]
        if generate_video:
            cmd.append("--generate-video")
        if cleanup_temp:
            cmd.append("--cleanup-temp")
        if self.config.sam3_checkpoint_path:
            cmd.extend(["--checkpoint-path", str(self.config.sam3_checkpoint_path)])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def get_chunk_progress(self, chunk_name: str) -> tuple[int, int]:
        """
        获取 chunk 处理进度

        Args:
            chunk_name: chunk 名称

        Returns:
            (processed_count, total_count)
        """
        chunk_dir = self.config.dataset_root / chunk_name
        mcap_dir = chunk_dir / "mcap"
        output_dir = self.config.sidewalk_root / chunk_name

        total = len(list(mcap_dir.glob("*.mcap"))) if mcap_dir.exists() else 0
        processed = len(list(output_dir.glob("*"))) - 1 if output_dir.exists() else 0
        return max(0, processed), total


# ========== 快捷函数接口 ==========

def create_processor(
    dataset_root: Optional[str | Path] = None,
    sidewalk_root: Optional[str | Path] = None,
    num_workers: int = 4,
    sample_fps: float = 20.0,
) -> SRNavProcessor:
    """
    创建处理器的快捷函数

    Args:
        dataset_root: 数据集根目录
        sidewalk_root: sidewalk 输出根目录
        num_workers: 并行 worker 数量
        sample_fps: 采样帧率

    Returns:
        SRNavProcessor: 处理器实例
    """
    config = ProcessingConfig(
        num_workers=num_workers,
        sample_fps=sample_fps,
    )
    if dataset_root:
        config.dataset_root = Path(dataset_root)
    if sidewalk_root:
        config.sidewalk_root = Path(sidewalk_root)
    return SRNavProcessor(config)


def process_mcap(
    mcap_path: str | Path,
    output_dir: Optional[str | Path] = None,
    sample_fps: float = 20.0,
) -> bool:
    """
    处理单个 MCAP 文件的快捷函数

    Args:
        mcap_path: MCAP 文件路径
        output_dir: 输出目录
        sample_fps: 采样帧率

    Returns:
        bool: 是否成功
    """
    processor = create_processor(sample_fps=sample_fps)
    result = processor.process_single_mcap(Path(mcap_path), Path(output_dir) if output_dir else None)
    return result.success


def process_chunk(
    chunk_name: str,
    num_workers: int = 4,
    skip_existing: bool = True,
) -> ProcessResult:
    """
    处理单个 chunk 的快捷函数

    Args:
        chunk_name: chunk 名称，如 "chunk_26"
        num_workers: 并行 worker 数量
        skip_existing: 跳过已处理的文件

    Returns:
        ProcessResult: 处理结果
    """
    processor = create_processor(num_workers=num_workers)
    return processor.process_chunk(chunk_name, skip_existing)


def batch_process_chunks(
    chunk_names: list[str],
    num_workers: int = 4,
    skip_existing: bool = True,
) -> list[ProcessResult]:
    """
    批量处理 chunks 的快捷函数

    Args:
        chunk_names: chunk 名称列表
        num_workers: 并行 worker 数量
        skip_existing: 跳过已处理的文件

    Returns:
        list[ProcessResult]: 处理结果列表
    """
    processor = create_processor(num_workers=num_workers)
    return processor.process_chunks(chunk_names, skip_existing)


if __name__ == "__main__":
    # 使用示例
    print("SR-Nav 数据处理模块")
    print("=" * 50)
    print("使用示例：")
    print("""
    # 方式 1: 使用类
    from srnav_processor import SRNavProcessor, ProcessingConfig

    config = ProcessingConfig(num_workers=4)
    processor = SRNavProcessor(config)
    result = processor.process_chunk("chunk_26")

    # 方式 2: 使用快捷函数
    from srnav_processor import process_chunk, batch_process_chunks

    # 处理单个 chunk
    result = process_chunk("chunk_26", num_workers=4)

    # 批量处理
    results = batch_process_chunks(["chunk_26", "chunk_27"], num_workers=4)

    # 处理单个 MCAP
    from srnav_processor import process_mcap
    success = process_mcap("/path/to/file.mcap")
    """)
