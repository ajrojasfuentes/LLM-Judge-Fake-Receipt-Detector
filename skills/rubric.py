"""
rubric.py

Prompt builder for receipt forgery forensic judges (VL Models).

This module loads:
- A system prompt template (Markdown) with placeholders
- A JSON output schema template
- A set of per-skill Markdown checklists

Expected placeholders in the system prompt template:
  {receipt_id}, {persona_name}, {persona_description},
  {skills}, {output_schema}, {evidence_pack}

Optional placeholders supported (safe to include even if unused):
  {analysis_date}, {timezone}

Design goals:
- Skill IDs are snake_case and align with output_schema["skill_results"] keys.
- Skills are inserted under an existing "###" heading, so this builder uses "####"
  (or deeper) headings for individual skills.
- Paths are resolved relative to templates_dir (never the current working directory),
  unless absolute paths are provided.

Usage:
    rubric = Rubric(templates_dir="templates")
    prompt = rubric.build_prompt(
        receipt_id="X00016469622",
        persona_name="Forensic Analyst",
        persona_description="You are ...",
        evidence_pack={"metadata": {...}},
    )
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


PathLike = Union[str, Path]
DEFAULT_TIMEZONE = "America/Bogota"


class RubricError(RuntimeError):
    """Raised for rubric loading/building errors."""


@dataclass(frozen=True)
class SkillSpec:
    """Definition of a single skill template."""
    skill_id: str          # snake_case ID used in JSON schema
    display_name: str      # human-friendly name for headings
    filename: str          # file name under templates_dir


# Default canonical skill order (matches your schema keys)
DEFAULT_SKILLS: Tuple[SkillSpec, ...] = (
    SkillSpec("math_consistency", "Math Consistency", "math_consistency.md"),
    SkillSpec("typography_analysis", "Typography Analysis", "typography_analysis.md"),
    SkillSpec("visual_authenticity", "Visual Authenticity", "visual_authenticity.md"),
    SkillSpec("layout_structure", "Layout Structure", "layout_structure.md"),
    SkillSpec("contextual_validation", "Contextual Validation", "contextual_validation.md"),
)


def _to_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _resolve_templates_dir(templates_dir: Optional[PathLike]) -> Path:
    """
    Resolve the templates directory in a predictable way.

    Precedence:
      1) explicit templates_dir argument
      2) environment variable RUBRIC_TEMPLATES_DIR
      3) <this_file_dir>/templates
    """
    if templates_dir is not None:
        base = _to_path(templates_dir).expanduser().resolve()
        if not base.is_dir():
            raise RubricError(f"templates_dir does not exist or is not a directory: {base}")
        return base

    env = os.getenv("RUBRIC_TEMPLATES_DIR")
    if env:
        base = Path(env).expanduser().resolve()
        if not base.is_dir():
            raise RubricError(f"RUBRIC_TEMPLATES_DIR does not exist or is not a directory: {base}")
        return base

    base = (Path(__file__).parent / "templates").expanduser().resolve()
    if not base.is_dir():
        raise RubricError(
            "Could not find templates directory. Provide templates_dir=... "
            "or set RUBRIC_TEMPLATES_DIR."
        )
    return base


def _read_text(path: Path, encoding: str = "utf-8") -> str:
    if not path.is_file():
        raise RubricError(f"Missing template file: {path}")
    return path.read_text(encoding=encoding).strip()


_HEADING_LEQ_3_RE = re.compile(r"^(#{1,3})\s+", re.MULTILINE)


def _ensure_no_headings_leq_3(markdown: str, *, where: str) -> None:
    """
    Skills are inserted under an existing '###' heading. To avoid breaking
    structure, reject skill markdown that contains '#', '##', or '###' headings.
    """
    m = _HEADING_LEQ_3_RE.search(markdown)
    if m:
        raise RubricError(
            f"Skill markdown contains a heading of level <= 3 in {where}. "
            "Use '####' or deeper, or remove headings."
        )


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False)


class Rubric:
    """
    Loads and compiles skill templates into a complete judge prompt.

    Notes:
    - If you change the output_schema.json keys under "skill_results", update
      DEFAULT_SKILLS or pass a custom skills=... list.
    """

    def __init__(
        self,
        templates_dir: Optional[PathLike] = None,
        system_prompt_filename: str = "system_prompt.md",
        output_schema_filename: str = "output_schema.json",
        skills: Sequence[SkillSpec] = DEFAULT_SKILLS,
        *,
        encoding: str = "utf-8",
        validate_skill_markdown: bool = True,
    ) -> None:
        self.templates_dir: Path = _resolve_templates_dir(templates_dir)
        self.encoding = encoding
        self.system_prompt_path = self.templates_dir / system_prompt_filename
        self.output_schema_path = self.templates_dir / output_schema_filename

        # Load templates
        self._system_prompt_template: str = _read_text(self.system_prompt_path, encoding=self.encoding)
        self._output_schema_text: str = _read_text(self.output_schema_path, encoding=self.encoding)

        try:
            self._output_schema_obj: Dict[str, Any] = json.loads(self._output_schema_text)
        except Exception as e:
            raise RubricError(f"Invalid JSON in output schema template: {self.output_schema_path}") from e

        # Load skills
        self._skill_specs: Tuple[SkillSpec, ...] = tuple(skills)
        self._skills: Dict[str, str] = {}
        for spec in self._skill_specs:
            skill_path = (self.templates_dir / spec.filename).expanduser().resolve()
            text = _read_text(skill_path, encoding=self.encoding)
            if validate_skill_markdown:
                _ensure_no_headings_leq_3(text, where=str(skill_path))
            self._skills[spec.skill_id] = text

        # Validate schema alignment (best-effort, but helpful)
        self._validate_schema_skill_keys()

    # ---------------------------
    # Public helpers
    # ---------------------------

    def available_skills(self) -> List[Tuple[str, str]]:
        """Return [(skill_id, display_name), ...] in canonical order."""
        return [(s.skill_id, s.display_name) for s in self._skill_specs]

    def get_output_schema_text(self) -> str:
        """Return the output schema template text (as loaded)."""
        return self._output_schema_text

    def get_output_schema(self) -> Dict[str, Any]:
        """Return the output schema template as a Python dict."""
        # Return a copy so callers can't mutate internal state accidentally.
        return json.loads(self._output_schema_text)

    # ---------------------------
    # Core builders
    # ---------------------------

    def build_skills_block(
        self,
        skill_ids: Optional[Sequence[str]] = None,
        *,
        header_level: int = 4,
        numbered: bool = True,
        include_ids_in_heading: bool = True,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Compile the skills into one Markdown block.

        Important: Skills are inserted under an existing "###" heading, so:
          header_level must be >= 4
        """
        if header_level < 4:
            raise ValueError("header_level must be >= 4 because skills are inserted under a '###' heading.")

        ordered_ids = [s.skill_id for s in self._skill_specs]
        chosen = list(skill_ids) if skill_ids is not None else ordered_ids

        unknown = [sid for sid in chosen if sid not in self._skills]
        if unknown:
            available = ", ".join(ordered_ids)
            raise RubricError(f"Unknown skill_ids: {unknown}. Available: {available}")

        # Keep canonical order even if a subset is provided
        chosen_set = set(chosen)
        chosen_in_order = [sid for sid in ordered_ids if sid in chosen_set]

        prefix = "#" * header_level
        blocks: List[str] = []

        for i, sid in enumerate(chosen_in_order, start=1):
            spec = next(s for s in self._skill_specs if s.skill_id == sid)
            title_parts = []
            if numbered:
                title_parts.append(f"{i}.")
            title_parts.append(spec.display_name)
            if include_ids_in_heading:
                title_parts.append(f"(`{sid}`)")
            heading = f"{prefix} " + " ".join(title_parts)
            blocks.append(f"{heading}\n\n{self._skills[sid].rstrip()}")

        return separator.join(blocks).rstrip() + "\n"

    def build_prompt(
        self,
        *,
        receipt_id: str,
        persona_name: str,
        persona_description: str,
        evidence_pack: Union[str, Mapping[str, Any], Sequence[Any]] = "",
        skill_ids: Optional[Sequence[str]] = None,
        analysis_date: Optional[str] = None,
        timezone: str = DEFAULT_TIMEZONE,
        header_level: int = 4,
        validate_schema_skills: bool = True,
    ) -> str:
        """
        Build the full prompt by formatting system_prompt.md.

        evidence_pack:
          - str: inserted as-is
          - dict/list: pretty-printed JSON for readability
        """
        skills_block = self.build_skills_block(
            skill_ids=skill_ids,
            header_level=header_level,
        )

        if isinstance(evidence_pack, str):
            evidence_text = evidence_pack.strip()
        else:
            evidence_text = _pretty_json(evidence_pack)

        if analysis_date is None:
            analysis_date = self._default_analysis_date(timezone)

        # Validate schema skill keys match what we're asking for (recommended)
        if validate_schema_skills:
            schema_ids = self._schema_skill_ids()
            requested_ids = self._requested_skill_ids(skill_ids)
            if schema_ids != requested_ids:
                raise RubricError(
                    "Mismatch between output_schema.json skill_results keys and requested skill_ids.\n"
                    f"schema skill_results keys: {sorted(schema_ids)}\n"
                    f"requested skill_ids:      {sorted(requested_ids)}\n"
                    "Fix by updating output_schema.json or the skills passed to Rubric."
                )

        try:
            return self._system_prompt_template.format(
                receipt_id=receipt_id,
                persona_name=persona_name,
                persona_description=persona_description,
                skills=skills_block,
                output_schema=self._output_schema_text,
                evidence_pack=evidence_text,
                analysis_date=analysis_date,  # optional placeholder
                timezone=timezone,            # optional placeholder
            ).rstrip() + "\n"
        except KeyError as e:
            # Make missing placeholder errors actionable
            raise RubricError(
                f"Missing placeholder in build_prompt() arguments: {e}. "
                "Check system_prompt.md placeholders."
            ) from e

    # ---------------------------
    # Internal validation
    # ---------------------------

    def _schema_skill_ids(self) -> set:
        skill_results = self._output_schema_obj.get("skill_results", {})
        if not isinstance(skill_results, dict):
            return set()
        return set(skill_results.keys())

    def _requested_skill_ids(self, skill_ids: Optional[Sequence[str]]) -> set:
        if skill_ids is None:
            return set(s.skill_id for s in self._skill_specs)
        return set(skill_ids)

    def _validate_schema_skill_keys(self) -> None:
        schema_ids = self._schema_skill_ids()
        if not schema_ids:
            # If schema is missing/invalid skill_results, don't hard-fail here,
            # but the build_prompt(validate_schema_skills=True) will catch mismatch.
            return

        #known_ids = set(s.skill_id for s in self._skill_specs)
        #missing_in_templates = sorted(schema_ids - known_ids)
        #extra_in_templates = sorted(known_ids - schema_ids)

        #if missing_in_templates or extra_in_templates:
            ## Non-fatal, but very useful in dev. Raise as error to avoid silent drift.
            #raise RubricError(
                #"output_schema.json and skill templates are out of sync.\n"
                #f"Missing templates for schema keys: {missing_in_templates}\n"
                #f"Extra templates not in schema:     {extra_in_templates}\n"
                #"Fix by updating DEFAULT_SKILLS or output_schema.json."
            #)

    def _default_analysis_date(self, timezone: str) -> str:
        """
        Return today's date in ISO format (YYYY-MM-DD) in the given timezone.
        If zoneinfo isn't available, fall back to local date.
        """
        if ZoneInfo is None:
            return date.today().isoformat()
        try:
            now = datetime.now(ZoneInfo(timezone))
            return now.date().isoformat()
        except Exception:
            return date.today().isoformat()