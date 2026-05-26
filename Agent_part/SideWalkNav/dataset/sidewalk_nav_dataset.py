"""SideWalk Nav Dataset (Refactored)

Modular design for SideWalk navigation dataset.
Full backward compatibility preserved, but with cleaner code structure.

Change Log:
    v4: Full refactoring, modular design
        - Separated Clip loading, frame sync, image loading, CoT management, System1 preparation, sample processing
    v3: Fixed FPV projection logic for trajectory generation
        - Fixed v coordinate formula: v = cy - fy*y_cam/z_cam (trajectory now goes UP)
        - First trajectory point forced to bottom center (image_width/2, image_height*0.9)
        - Trajectory sampling starts from current_frame_idx + 1 (avoids x_robot=0 issue)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset

from .rope2d import get_rope_index_2, get_rope_index_25
from .sidewalk_trajectory_generator import (
    CameraParams,
    TrajectoryConfig,
    SidewalkTrajectoryGenerator,
)
from .sidewalk_clip import SidewalkClip, SidewalkClipLoader
from .sidewalk_frame_syncer import SidewalkFrameSyncer
from .sidewalk_image_loader import SidewalkImageLoader, SidewalkImageProcessor
from .sidewalk_cot_manager import StructuredCoTManager, StructuredCoTConfig
from .sidewalk_system1_preparer import (
    System1Preparer, System1Config,
    quat_to_yaw, get_trajectory_relative_to_frame,
    interpolate_and_resample_trajectory
)
from .sidewalk_sample_processor import (
    SampleProcessor, create_sample_processor, SampleProcessResult
)
from .sidewalk_collator import SideWalkDataCollator
from .utils import rank0_print, set_local_rank
from ..prompt.sidewalk import (
    DEFAULT_SIDEWALK_INSTRUCTION,
    DEFAULT_SIDEWALK_PROMPT_TEMPLATE,
    DEFAULT_SIDEWALK_SYSTEM_PROMPT,
    build_sidewalk_prompt,
    format_pixel_trajectory_response,
    format_structured_cot_response,
)


# ========== Backward compatibility section ==========
# Keep these functions and classes to avoid breaking existing code

# Global rank (kept in sync with utils module)
local_rank = None


def _update_local_rank(rank: Optional[int]):
    """Update both global variable and utils module rank."""
    global local_rank
    local_rank = rank
    set_local_rank(rank)


# Backward compatibility versions of these functions
def quat_to_yaw_backward(quat: np.ndarray) -> float:
    return quat_to_yaw(quat)


def get_trajectory_relative_to_frame_backward(
    poses: np.ndarray,
    current_pos: np.ndarray,
    current_quat: np.ndarray,
) -> np.ndarray:
    return get_trajectory_relative_to_frame(poses, current_pos, current_quat)


def interpolate_and_resample_trajectory_backward(
    traj: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    return interpolate_and_resample_trajectory(traj, num_samples)


def rank0_print_backward(*args, **kwargs):
    rank0_print(*args, **kwargs)


# Redirect to new module, but keep the names
quat_to_yaw = quat_to_yaw_backward
get_trajectory_relative_to_frame = get_trajectory_relative_to_frame_backward
interpolate_and_resample_trajectory = interpolate_and_resample_trajectory_backward
rank0_print = rank0_print_backward


# ========== Configuration class ==========

@dataclass
class SideWalkDataConfig:
    """SideWalk Dataset configuration.

    Attributes:
        use_bev: Whether to include BEV as second image
        use_mp4_directly: Whether to read directly from MP4 instead of jpg
        num_future_steps: Number of future steps to predict
        sample_step: Step between samples
        num_trajectory_points: Number of points in output trajectory
        use_trajectory: Whether to output trajectory instead of single point
        use_system1: Whether to enable System1 (trajectory diffusion) training
        n_query: Number of latent queries for System1
        system1_sample_step: Step between frames for System1 trajectory sampling
        system1_num_frames: Number of frames to include in System1 trajectory
        predict_step_nums: Number of steps to predict for System1
        prompt_template: Prompt template used for training
        use_structured_cot: Whether to mix Phase-1 structured CoT text targets
        structured_cot_root: Optional external root for sidewalk_cot_v1 JSON files
        structured_cot_sample_ratio: Deterministic ratio of samples using CoT targets
        structured_cot_missing_policy: "fallback" or "error" when JSON is missing
        use_system_prompt: Whether to prepend the canonical SideWalk system role
        system_prompt: Optional override for the canonical SideWalk system role
    """
    use_bev: bool = True
    use_mp4_directly: bool = True
    num_future_steps: int = 10
    sample_step: int = 5
    num_trajectory_points: int = 16
    use_trajectory: bool = True
    use_system1: bool = False
    n_query: int = 16
    system1_sample_step: int = 2
    system1_num_frames: int = 12
    predict_step_nums: int = 32
    prompt_template: str = DEFAULT_SIDEWALK_PROMPT_TEMPLATE
    use_structured_cot: bool = False
    structured_cot_root: Optional[str] = None
    structured_cot_sample_ratio: float = 1.0
    structured_cot_missing_policy: str = "fallback"
    use_system_prompt: bool = True
    system_prompt: Optional[str] = None


# ========== Backward compatibility: preprocess_sidewalk_conversation ==========

def preprocess_sidewalk_conversation(
    sources: List[List[Dict[str, str]]],
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Preprocess multi-turn conversation data (backward compatible).

    Args:
        sources: List of conversation histories
        tokenizer: HuggingFace tokenizer
        grid_thw_image: Grid dimensions for image tokens

    Returns:
        Dictionary with 'input_ids' and 'labels'
    """
    conv = sources[0]

    # Build prompt with proper image token placement
    prompt_parts = []
    for turn in conv:
        role = turn["from"]
        value = turn["value"]

        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{value}<|im_end|>\n")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{value}<|im_end|>\n")
        elif role == "assistant":
            # Stop at assistant turn - this is what we need to predict
            prompt_parts.append(f"<|im_start|>assistant\n")
            break

    # Tokenize prompt and target
    full_prompt = "".join(prompt_parts)
    input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)

    # Get the assistant response for labels
    assistant_turns = [t for t in conv if t["from"] == "assistant"]
    if assistant_turns:
        target_text = assistant_turns[0]["value"]
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        target_ids.append(tokenizer.eos_token_id)
    else:
        target_ids = []

    # Combine input and target
    full_input_ids = input_ids + target_ids
    labels = [-100] * len(input_ids) + target_ids  # Mask input part

    return {
        "input_ids": torch.tensor([full_input_ids]),
        "labels": torch.tensor([labels]),
    }


# ========== Main Dataset class ==========

class SideWalkNavDataset(Dataset):
    """SideWalk Nav Dataset.

    Implements functionality by composing modules instead of having all logic in one class.
    """

    def __init__(
        self,
        data_root: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: Any,
        dataset_name: Optional[str] = None,  # Not used, kept for backward compatibility
        data_config: Optional[SideWalkDataConfig] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.config = data_config or SideWalkDataConfig()

        # Initialize all components
        self._init_components()

        # Load clips
        self.clips = self._load_all_clips()
        self._build_index()
        rank0_print(f"Loaded {len(self.clips)} clips, {len(self.index)} samples")

    def _init_components(self):
        """Initialize all sub-components."""
        # RoPE index setup
        self.model_type = getattr(self.data_args, "model_type", "qwen2.5vl")
        if self.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        # Configure image processor
        if hasattr(self.data_args, "image_processor"):
            self.data_args.image_processor.max_pixels = getattr(
                self.data_args, "max_pixels", 1280 * 28 * 28
            )
            self.data_args.image_processor.min_pixels = getattr(
                self.data_args, "min_pixels", 256 * 28 * 28
            )

        # Trajectory generator
        self.trajectory_config = TrajectoryConfig(
            num_trajectory_points=self.config.num_trajectory_points,
            num_future_steps=self.config.num_future_steps,
        )
        self.trajectory_generator = SidewalkTrajectoryGenerator(config=self.trajectory_config)

        # CoT manager
        self.cot_config = StructuredCoTConfig(
            use_structured_cot=self.config.use_structured_cot,
            structured_cot_root=self.config.structured_cot_root,
            structured_cot_sample_ratio=self.config.structured_cot_sample_ratio,
            structured_cot_missing_policy=self.config.structured_cot_missing_policy,
        )
        self.cot_manager = StructuredCoTManager(self.cot_config)

        # System1 preparer
        self.system1_preparer = None
        if self.config.use_system1:
            system1_config = System1Config(
                n_query=self.config.n_query,
                system1_sample_step=self.config.system1_sample_step,
                system1_num_frames=self.config.system1_num_frames,
                predict_step_nums=self.config.predict_step_nums,
            )
            self.system1_preparer = System1Preparer(system1_config)

        # Sample processor (strategy pattern)
        self.sample_processor = self._create_sample_processor()

    def _create_sample_processor(self) -> SampleProcessor:
        """Create appropriate sample processor."""
        processor_kwargs = {'data_args': self.data_args}
        if self.model_type in ['qwen2.5vl', 'qwen2vl']:
            processor_kwargs['processor'] = getattr(self.data_args, 'processor', None)
        else:
            processor_kwargs['tokenizer'] = self.tokenizer
            processor_kwargs['get_rope_index'] = self.get_rope_index
        return create_sample_processor(self.model_type, **processor_kwargs)

    def _load_all_clips(self) -> List[SidewalkClip]:
        """Load all clips."""
        clips = []
        # Look for chunk directories
        chunk_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("chunk_")]
        if not chunk_dirs:
            chunk_dirs = [self.data_root]

        for chunk_dir in chunk_dirs:
            for clip_dir in [d for d in chunk_dir.iterdir() if d.is_dir()]:
                clip = SidewalkClipLoader.load(clip_dir, self.config.use_mp4_directly)
                if clip is not None:
                    clips.append(clip)
        return clips

    def _build_index(self):
        """Build sample index."""
        self.index = []
        for clip_idx, clip in enumerate(self.clips):
            for frame_idx in range(0, clip.num_frames - self.config.num_future_steps, self.config.sample_step):
                if self.config.use_bev and not SidewalkFrameSyncer.has_bev_for_frame(clip, frame_idx):
                    continue
                self.index.append({
                    "clip_idx": clip_idx,
                    "frame_idx": frame_idx,
                })

    def __len__(self) -> int:
        return len(self.index)

    def _get_item_impl(self, idx: int) -> Dict[str, torch.Tensor]:
        """Internal implementation for getting a sample."""
        index_info = self.index[idx]
        clip_idx = index_info["clip_idx"]
        frame_idx = index_info["frame_idx"]
        clip = self.clips[clip_idx]

        # 1. Load images
        fpv_image = SidewalkImageLoader.get_fpv_image(clip, frame_idx)
        bev_image = None
        if self.config.use_bev:
            bev_image = SidewalkImageLoader.get_bev_image(clip, frame_idx)

        # 2. Get target trajectory/point
        pixel_trajectory = None
        u_norm, v_norm = 0.5, 0.5
        if self.config.use_trajectory:
            if clip.camera_params is not None:
                self.trajectory_generator.set_camera_params(clip.camera_params)
                pixel_trajectory = self.trajectory_generator.generate_trajectory(
                    clip.positions, clip.orientations, frame_idx
                )
                u_norm, v_norm = float(pixel_trajectory[0, 0]), float(pixel_trajectory[0, 1])
            else:
                pixel_trajectory = self.trajectory_generator.generate_dummy_trajectory()
                u_norm, v_norm = 0.5, 0.9
        else:
            if clip.camera_params is not None:
                self.trajectory_generator.set_camera_params(clip.camera_params)
                u, v = self.trajectory_generator.project_map_to_fpv(
                    clip.positions[frame_idx, :2],
                    clip.orientations[frame_idx],
                    clip.positions[min(frame_idx + self.config.num_future_steps, clip.num_frames - 1), :2],
                )
                u_norm = u / clip.camera_params.image_width
                v_norm = v / clip.camera_params.image_height

        image_width = clip.camera_params.image_width if clip.camera_params is not None else 1920
        image_height = clip.camera_params.image_height if clip.camera_params is not None else 1080

        # 3. Build prompt
        instruction = getattr(self.data_args, "sidewalk_instruction", DEFAULT_SIDEWALK_INSTRUCTION) or DEFAULT_SIDEWALK_INSTRUCTION
        system_prompt = None
        if self.config.use_system_prompt:
            system_prompt = self.config.system_prompt or getattr(self.data_args, "sidewalk_system_prompt", DEFAULT_SIDEWALK_SYSTEM_PROMPT) or None

        prompt_num_points = self.config.num_trajectory_points if self.config.use_trajectory else 1
        prompt_template = getattr(self.data_args, "prompt_template", self.config.prompt_template)

        # 4. Process Structured CoT
        structured_cot_used = False
        structured_cot_path = None
        if self.config.use_trajectory and pixel_trajectory is not None:
            trajectory_response = format_pixel_trajectory_response(pixel_trajectory, image_width, image_height)
        else:
            pixel_x = int(u_norm * image_width)
            pixel_y = int(v_norm * image_height)
            trajectory_response = f"{pixel_x} {pixel_y}"
        response_text = trajectory_response

        clip_name = clip.clip_dir.name
        if self.cot_manager.should_use_for_sample(clip_name, frame_idx):
            annotation, path = self.cot_manager.load_annotation(clip.clip_dir, frame_idx)
            if annotation is not None:
                prompt_template = "structured_cot_trajectory"
                response_text = format_structured_cot_response(annotation, fallback_trajectory_text=trajectory_response)
                structured_cot_used = True
                structured_cot_path = str(path) if path else None
        elif prompt_template == "structured_cot_trajectory":
            response_text = format_structured_cot_response({}, fallback_trajectory_text=trajectory_response)

        prompt_with_images = build_sidewalk_prompt(
            instruction=instruction, num_points=prompt_num_points,
            prompt_template=prompt_template, include_image_tokens=True
        )
        prompt_without_images = build_sidewalk_prompt(
            instruction=instruction, num_points=prompt_num_points,
            prompt_template=prompt_template, include_image_tokens=False
        )

        # 5. System1 data preparation
        system1_data = None
        if self.system1_preparer is not None:
            system1_data = self.system1_preparer.prepare(clip, frame_idx)

        # 6. Process sample with strategy
        extra_context = {
            'use_bev': self.config.use_bev,
            'instruction': instruction,
            'structured_cot_used': structured_cot_used,
            'structured_cot_path': structured_cot_path,
            'system1_data': system1_data,
            'image_width': image_width,
            'image_height': image_height,
        }

        result = self.sample_processor.process(
            fpv_image=fpv_image,
            bev_image=bev_image,
            prompt_text_with_images=prompt_with_images,
            prompt_text_without_images=prompt_without_images,
            response_text=response_text,
            u_norm=u_norm,
            v_norm=v_norm,
            pixel_trajectory=pixel_trajectory,
            system_prompt=system_prompt,
            extra_context=extra_context,
        )

        # 7. Assemble final result
        output_dict = {
            "input_ids": result.input_ids,
            "labels": result.labels,
            "pixel_values": result.pixel_values,
            "image_grid_thw": result.image_grid_thw,
            "pixel_coords": result.pixel_coords,
        }
        if result.attention_mask is not None:
            output_dict["attention_mask"] = result.attention_mask
        if result.regression_token_index is not None:
            output_dict["regression_token_index"] = result.regression_token_index
        if result.pixel_trajectory is not None:
            output_dict["pixel_trajectory"] = result.pixel_trajectory
        if result.structured_cot_used is not None:
            output_dict["structured_cot_used"] = result.structured_cot_used
        if result.structured_cot_path is not None:
            output_dict["structured_cot_path"] = result.structured_cot_path
        if result.position_ids is not None:
            output_dict["position_ids"] = result.position_ids
        if result.extra_data is not None:
            output_dict.update(result.extra_data)

        return output_dict

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        num_retries = 3
        for attempt in range(num_retries):
            try:
                return self._get_item_impl(idx)
            except Exception as e:
                if attempt < num_retries - 1:
                    rank0_print(f"Attempt {attempt + 1} failed for idx {idx}: {e}")
                    time.sleep(0.1)
                else:
                    raise
        return self._get_item_impl(idx)

    def close(self):
        """Clean up resources."""
        for clip in self.clips:
            clip.close_video()

    # ========== Backward compatibility methods (not recommended for use) ==========

    def _get_fpv_pixel_goal(self, clip_info: Dict, current_frame_idx: int) -> Tuple[float, float]:
        """Backward compatible: Get target pixel coordinates (not recommended)."""
        # This method is for compatibility with old code, new code should use new modules
        positions = clip_info.get("positions")
        orientations = clip_info.get("orientations")
        camera_params = clip_info.get("camera_params")

        if camera_params is None:
            return 0.5, 0.5

        current_pos = positions[current_frame_idx, :2]
        current_quat = orientations[current_frame_idx]
        future_idx = min(current_frame_idx + self.config.num_future_steps, len(positions) - 1)
        future_pos = positions[future_idx, :2]

        self.trajectory_generator.set_camera_params(camera_params)
        u, v = self.trajectory_generator.project_map_to_fpv(current_pos, current_quat, future_pos)
        u_norm = u / camera_params.image_width
        v_norm = v / camera_params.image_height
        return u_norm, v_norm

    def _get_fpv_pixel_trajectory(self, clip_info: Dict, current_frame_idx: int) -> np.ndarray:
        """Backward compatible: Get target pixel trajectory (not recommended)."""
        positions = clip_info.get("positions")
        orientations = clip_info.get("orientations")
        camera_params = clip_info.get("camera_params")

        if camera_params is None:
            return self.trajectory_generator.generate_dummy_trajectory()

        self.trajectory_generator.set_camera_params(camera_params)
        return self.trajectory_generator.generate_trajectory(positions, orientations, current_frame_idx)

    # ========== Backward compatibility placeholder methods (deprecated) ==========
    def _load_single_clip(self, clip_dir: Path) -> Optional[Dict]:
        """Deprecated, use SidewalkClipLoader instead."""
        raise NotImplementedError("Use SidewalkClipLoader instead")

    def _load_bev_image_paths(self, clip_dir: Path) -> List[Path]:
        """Deprecated, use SidewalkClipLoader instead."""
        raise NotImplementedError("Use SidewalkClipLoader instead")

    def _frame_time_seconds(self, clip_info: Dict, frame_idx: int) -> Optional[float]:
        """Deprecated, use SidewalkFrameSyncer instead."""
        raise NotImplementedError("Use SidewalkFrameSyncer instead")

    def _estimate_fpv_fps(self, clip_info: Dict) -> float:
        """Deprecated, use SidewalkFrameSyncer instead."""
        raise NotImplementedError("Use SidewalkFrameSyncer instead")

    def _map_fpv_frame_to_bev_index(self, clip_info: Dict, frame_idx: int) -> Optional[int]:
        """Deprecated, use SidewalkFrameSyncer instead."""
        raise NotImplementedError("Use SidewalkFrameSyncer instead")

    def _has_bev_for_frame(self, clip_info: Dict, frame_idx: int) -> bool:
        """Deprecated, use SidewalkFrameSyncer instead."""
        raise NotImplementedError("Use SidewalkFrameSyncer instead")

    def _get_fpv_image(self, clip_info: Dict, frame_idx: int) -> Any:
        """Deprecated, use SidewalkImageLoader instead."""
        raise NotImplementedError("Use SidewalkImageLoader instead")

    def _get_bev_image(self, clip_info: Dict, frame_idx: int) -> Any:
        """Deprecated, use SidewalkImageLoader instead."""
        raise NotImplementedError("Use SidewalkImageLoader instead")

    def _structured_cot_candidate_paths(self, clip_info: Dict, frame_idx: int) -> List[Path]:
        """Deprecated, use StructuredCoTManager instead."""
        raise NotImplementedError("Use StructuredCoTManager instead")

    def _load_structured_cot_annotation(self, clip_info: Dict, frame_idx: int) -> Tuple:
        """Deprecated, use StructuredCoTManager instead."""
        raise NotImplementedError("Use StructuredCoTManager instead")

    def _should_use_structured_cot(self, clip_info: Dict, frame_idx: int) -> bool:
        """Deprecated, use StructuredCoTManager instead."""
        raise NotImplementedError("Use StructuredCoTManager instead")

    def _get_traj_images(self, clip_info: Dict, current_frame_idx: int, num_frames: int, sample_step: int) -> Tuple:
        """Deprecated, use SidewalkImageLoader instead."""
        raise NotImplementedError("Use SidewalkImageLoader instead")

    def _get_traj_poses(self, clip_info: Dict, current_frame_idx: int, traj_frame_indices: List[int], predict_step_nums: int) -> np.ndarray:
        """Deprecated, use System1Preparer instead."""
        raise NotImplementedError("Use System1Preparer instead")

    def _project_with_camera_params(self, dx: float, dy: float, dz: float, current_pos: np.ndarray, current_ori: np.ndarray, K: np.ndarray, T: np.ndarray) -> Tuple[int, int]:
        """Deprecated placeholder method."""
        raise NotImplementedError("This method is deprecated")

    def _process_single_image(self, image: Any) -> Tuple:
        """Deprecated, use SidewalkImageProcessor instead."""
        raise NotImplementedError("Use SidewalkImageProcessor instead")

    def _build_conversation(self, pixel_x: int, pixel_y: int, instruction: Optional[str] = None, prompt_text: Optional[str] = None, response_text: Optional[str] = None, system_prompt: Optional[str] = None) -> List[Dict]:
        """Deprecated, use SampleProcessor instead."""
        raise NotImplementedError("Use SampleProcessor instead")


# ========== make_supervised_sidewalk_data_module ==========

def make_supervised_sidewalk_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: Any,
) -> Dict[str, Any]:
    """Create data module for SideWalk Nav training.

    Args:
        tokenizer: HuggingFace tokenizer
        data_args: Data configuration

    Returns:
        Dictionary with 'train_dataset', 'eval_dataset', 'data_collator'
    """
    data_config = SideWalkDataConfig(
        use_bev=getattr(data_args, "use_bev", True),
        use_mp4_directly=getattr(data_args, "use_mp4_directly", False),
        num_future_steps=getattr(data_args, "num_future_steps", 10),
        sample_step=getattr(data_args, "sample_step", 5),
        num_trajectory_points=getattr(data_args, "num_trajectory_points", 16),
        use_trajectory=getattr(data_args, "use_trajectory", True),
        use_system1=getattr(data_args, "use_system1", False) or (getattr(data_args, "system1", None) is not None),
        n_query=getattr(data_args, "n_query", 16),
        system1_sample_step=getattr(data_args, "system1_sample_step", 2),
        system1_num_frames=getattr(data_args, "system1_num_frames", 12),
        predict_step_nums=getattr(data_args, "predict_step_nums", 32),
        prompt_template=getattr(data_args, "prompt_template", DEFAULT_SIDEWALK_PROMPT_TEMPLATE),
        use_structured_cot=getattr(data_args, "use_structured_cot", False),
        structured_cot_root=getattr(data_args, "structured_cot_root", None),
        structured_cot_sample_ratio=getattr(data_args, "structured_cot_sample_ratio", 1.0),
        structured_cot_missing_policy=getattr(data_args, "structured_cot_missing_policy", "fallback"),
        use_system_prompt=getattr(data_args, "use_sidewalk_system_prompt", True),
        system_prompt=getattr(data_args, "sidewalk_system_prompt", None),
    )

    train_dataset = SideWalkNavDataset(
        data_root=getattr(data_args, "data_root", "./data"),
        tokenizer=tokenizer,
        data_args=data_args,
        data_config=data_config,
    )

    data_collator = SideWalkDataCollator(tokenizer=tokenizer, data_args=data_args)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }
