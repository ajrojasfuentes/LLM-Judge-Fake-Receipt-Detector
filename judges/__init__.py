from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from .base_judge import BaseJudge, JudgeResult
from .glm_judge import GLMJudge
from .qwen_judge import QwenJudge
from .voting import FinalVerdict, VotingEngine


def load_judges_from_config(config_path: str | Path = "configs/judges.yaml") -> List[BaseJudge]:
    cfg_path = Path(config_path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    judges_cfg = cfg.get("judges", [])
    judges: List[BaseJudge] = []

    for jc in judges_cfg:
        model = str(jc.get("model", "")).strip()
        base_kwargs = dict(
            judge_id=jc["id"],
            judge_name=jc["name"],
            persona_description=str(jc.get("persona", "")).strip(),
            temperature=float(jc.get("temperature", 0.3)),
            max_tokens=int(jc.get("max_tokens", 1024)),
            focus_skills=jc.get("focus_skills"),
            model_id=model or None,
        )

        if "qwen" in model.lower():
            judges.append(QwenJudge(**base_kwargs))
        elif "glm" in model.lower() or "zai-org" in model.lower():
            judges.append(GLMJudge(**base_kwargs))
        else:
            raise ValueError(f"Unsupported judge model in config: {model}")

    return judges


__all__ = [
    "BaseJudge",
    "JudgeResult",
    "QwenJudge",
    "GLMJudge",
    "VotingEngine",
    "FinalVerdict",
    "load_judges_from_config",
]
