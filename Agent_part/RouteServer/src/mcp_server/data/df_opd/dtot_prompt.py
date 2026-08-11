"""RouteServer-owned visible prompt contract for the D-to-T baseline."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Mapping, Sequence

DF_OPD_DTOT_PROMPT_POLICY_ID = "urbannav_df_opd_dtot_prompt_v1"
DF_OPD_DTOT_DECISION_SYSTEM_PROMPT = (
    "You are the short-horizon decision component of an assistive urban navigation model.\n"
    "Use only the prompt-visible FPV, history, trusted instruction, and BEV evidence supplied "
    "in this example. FPV is authoritative for immediate visible safety. A prompt-visible "
    "BEV route overlay may support route direction and upcoming topology when its orientation "
    "is clear, but it cannot prove local traversability, clearance, or collision safety.\n\n"
    "Return exactly one object with exactly these three JSON keys and no others:\n"
    "<Decision>{\"selected_action_label\":\"<one supplied action_labels value>\",\"speed\":\"<normal|slow|stop>\",\"response\":\"<proceed|slow|yield|detour|stop>\"}</Decision>\n\n"
    "The tuple must be coherent: selected_action_label=stop requires speed=stop and "
    "response=stop. A moving action cannot use speed=stop or response=stop. response=slow, "
    "yield, or detour requires speed=slow. Do not output a think block, reasoning sections, "
    "Finish, trajectory, private labels, markdown, or text outside the single tag.")
DF_OPD_DTOT_PROMPT_POLICY_DIGEST = hashlib.sha256(
    DF_OPD_DTOT_DECISION_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
_QUERY_PATTERN = re.compile(r"\[TRAJ(?:_\d+)?\]")
_DECISION_TAG = "<TASK_DECISION>"
_DIRECT_LATENT_TAG = "<TASK_TRAJECTORY_CFM_DIRECT_LATENT>"
_USER_OUTPUT_INSTRUCTION = (
    "Return exactly one standalone <Decision> object with keys "
    "selected_action_label, speed, and response in that order.")


def build_df_opd_dtot_student_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compile direct-latent source messages into a Decision-visible D-to-T view."""

    if not isinstance(messages, Sequence) or not messages:
        raise ValueError("DF-OPD D-to-T prompt requires non-empty messages")
    result = [_transform_value(dict(message)) for message in copy.deepcopy(messages)]
    system_index = next((index for index, message in enumerate(result)
                         if message.get("role") == "system"), None)
    if system_index is not None:
        result[system_index]["content"] = DF_OPD_DTOT_DECISION_SYSTEM_PROMPT
    user_index = next((index for index in range(len(result) - 1, -1, -1)
                       if result[index].get("role") == "user"), None)
    if user_index is None:
        raise ValueError("DF-OPD D-to-T prompt requires a user message")
    content = result[user_index].get("content")
    if isinstance(content, str):
        content = content.replace(_DIRECT_LATENT_TAG, _DECISION_TAG).rstrip()
        if _USER_OUTPUT_INSTRUCTION not in content:
            content = f"{content}\n{_USER_OUTPUT_INSTRUCTION}"
        result[user_index]["content"] = content
    elif isinstance(content, list):
        content = list(content)
        content.append({"type": "text", "text": _USER_OUTPUT_INSTRUCTION})
        result[user_index]["content"] = content
    else:
        raise ValueError("DF-OPD D-to-T user content must be text or a content list")
    return result


def _transform_value(value: Any) -> Any:
    """Remove prompt-side queries recursively from copied message content."""

    if isinstance(value, str):
        return _QUERY_PATTERN.sub("", value)
    if isinstance(value, list):
        return [_transform_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _transform_value(item) for key, item in value.items()}
    return value


__all__ = [
    "DF_OPD_DTOT_DECISION_SYSTEM_PROMPT",
    "DF_OPD_DTOT_PROMPT_POLICY_DIGEST",
    "DF_OPD_DTOT_PROMPT_POLICY_ID",
    "build_df_opd_dtot_student_messages",
]

