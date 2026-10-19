"""MCP endpoints for vision tools."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import Context
from mcp_server.llm import analyze_image_with_llm, analyze_local_image_with_llm
from mcp_server.services.fpv_segmentation_service import segment_fpv_frame as segment_fpv_frame_service
from mcp_server.mcp_api.app import mcp

@mcp.tool()
async def analyze_image(
    image_url: str,
    ctx: Context,
    question: str = "What is in this image?",
    detail: str = "auto",
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze an image with the configured multimodal LLM endpoint.

    This tool sends an image URL and an optional question to the configured LLM
    provider (for example Volcengine ARK / Doubao or another OpenAI-compatible
    multimodal endpoint) and returns a concise text answer.

    Args:
        image_url: Publicly accessible image URL
        question: Prompt or question about the image
        detail: Vision detail level ("auto", "low", "high")
        system_prompt: Optional system instruction override

    Returns:
        Provider/model metadata and the model's answer
    """
    await ctx.info(f"Analyzing image with detail={detail}")
    return await analyze_image_with_llm(
        image_url=image_url,
        question=question,
        detail=detail,
        system_prompt=system_prompt,
    )


@mcp.tool()
async def analyze_local_image(
    image_path: str,
    ctx: Context,
    question: str = "What is in this image?",
    detail: str = "auto",
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a local image file with the configured multimodal LLM endpoint.

    This tool reads a local image path, converts it into an inline data URL,
    and sends it to the configured multimodal model. It is useful for local
    FPV / PoV experiments where the image is not publicly hosted.

    Args:
        image_path: Absolute or user-resolvable local image path
        question: Prompt or question about the image
        detail: Vision detail level ("auto", "low", "high")
        system_prompt: Optional system instruction override

    Returns:
        Provider/model metadata and the model's answer
    """
    await ctx.info(f"Analyzing local image with detail={detail}")
    return await analyze_local_image_with_llm(
        image_path=image_path,
        question=question,
        detail=detail,
        system_prompt=system_prompt,
    )


@mcp.tool()
async def segment_fpv_frame(
    ctx: Context,
    sequence_name: str,
    sam3_output_base_dir: str,
    image_path: Optional[str] = None,
    frame_name: Optional[str] = None,
    camera_view: str = "camera_front",
    run_sam3: bool = False,
    sam3_env_name: str = "sam3",
    sam3_script: str = "/home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/prepare_frames_and_run_batch_sam3.py",
    rides_base_dir: Optional[str] = None,
    frames_base_dir: Optional[str] = None,
    recordings_subdir: str = "recordings",
    stream_pattern: str = "*_video.m3u8",
    fps: float = 1.0,
    gpus: Optional[List[int]] = None,
    min_score: float = 0.3,
    finetuned: bool = True,
    checkpoint_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    overwrite_frames: bool = False,
    provider: str = "sam3",
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build or load SAM3 segmentation artifacts for an FPV frame.

    This tool supports two modes:
    - artifact lookup only: run_sam3=false, expects existing SAM3 outputs
    - full execution: run_sam3=true, launches external SAM3 pipeline first
    """
    await ctx.info(
        "Resolving SAM3 segmentation artifacts "
        f"sequence={sequence_name} camera={camera_view} run_sam3={run_sam3}"
    )
    return await segment_fpv_frame_service(
        sequence_name=sequence_name,
        sam3_output_base_dir=sam3_output_base_dir,
        image_path=image_path,
        frame_name=frame_name,
        camera_view=camera_view,
        run_sam3=run_sam3,
        sam3_env_name=sam3_env_name,
        sam3_script=sam3_script,
        rides_base_dir=rides_base_dir,
        frames_base_dir=frames_base_dir,
        recordings_subdir=recordings_subdir,
        stream_pattern=stream_pattern,
        fps=fps,
        gpus=gpus,
        min_score=min_score,
        finetuned=finetuned,
        checkpoint_path=checkpoint_path,
        max_frames=max_frames,
        overwrite_frames=overwrite_frames,
        provider=provider,
        request_id=request_id,
    )
