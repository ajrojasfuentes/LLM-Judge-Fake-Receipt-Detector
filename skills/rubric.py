"""
rubric.py

Prompt builder for receipt forgery forensic judges (VL Models).

This module loads:
- A system prompt template (Markdown) with placeholders
- A JSON output schema *template* (example JSON object shown to the model)
- A catalog of per-skill Markdown checklists

Expected placeholders in the system prompt template:
  {receipt_id}, {persona_name}, {persona_description},
  {skills}, {output_schema}, {evidence_pack}

Optional placeholders supported (safe to include even if unused):
  {analysis_date}, {timezone}

Key behavior (updated):
- DEFAULT_SKILLS is a *catalog of available skills* (options), not the selection.
- build_prompt(..., skill_ids=[...]) selects a subset of skills to insert into
  the prompt and dynamically *prunes* output_schema["skill_results"] so it only
  contains those selected skills.
- The builder also patches the system prompt constraint line about "skill_results"
  (if present) so it matches the selected skills.

Design goals:
- Skill IDs are snake_case and align with output_schema["skill_results"] keys.
- Skills are inserted under an existing "###" heading, so this builder uses "####"
  (or deeper) headings for individual skills.
- Paths are resolved relative to templates_dir (never the current working directory),
  unless absolute paths are provided.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
    skill_id: str          # snake_case ID used in JSON output
    display_name: str      # human-friendly name for headings
    filename: str          # file name under templates_dir


# Catalog of available skills (in canonical order).
# NOTE: This is *not* the selection; selection is provided via skill_ids=...
DEFAULT_SKILLS: Dict[str, SkillSpec] = {
    "math_consistency": SkillSpec("math_consistency", "Math Consistency", "math_consistency.md"),
    "typography_analysis": SkillSpec("typography_analysis", "Typography Analysis", "typography_analysis.md"),
    "visual_authenticity": SkillSpec("visual_authenticity", "Visual Authenticity", "visual_authenticity.md"),
    "layout_structure": SkillSpec("layout_structure", "Layout Structure", "layout_structure.md"),
    "contextual_validation": SkillSpec("contextual_validation", "Contextual Validation", "contextual_validation.md"),
}

# Backwards-compatible default list (old API expected a Sequence[SkillSpec])
DEFAULT_SKILL_SPECS: Tuple[SkillSpec, ...] = tuple(DEFAULT_SKILLS.values())


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

# Matches the constraint line that currently hard-codes "all 5 skills"
# Example in template: 4. `"skill_results"` MUST include **all 5 skills** ...
_SKILL_RESULTS_CONSTRAINT_RE = re.compile(
    r"^\s*4\.\s+.*\"skill_results\".*$",
    re.MULTILINE,
)


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


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class Rubric:
    """
    Loads and compiles skill templates into a complete judge prompt.

    Notes:
    - output_schema.json is treated as a *base template* (usually containing all
      skill_results keys). build_prompt(...) prunes it to the requested skill_ids.
    """

    def __init__(
        self,
        templates_dir: Optional[PathLike] = None,
        system_prompt_filename: str = "system_prompt.md",
        output_schema_filename: str = "output_schema.json",
        skills: Optional[Sequence[SkillSpec]] = None,
        skills_catalog: Optional[Mapping[str, SkillSpec]] = None,
        *,
        encoding: str = "utf-8",
        validate_skill_markdown: bool = True,
    ) -> None:
        """
        Backwards compatible:
          - Old API: Rubric(..., skills=[SkillSpec(...), ...])
          - New API: Rubric(..., skills_catalog={skill_id: SkillSpec(...), ...})

        If both are provided, raises.
        """
        if skills is not None and skills_catalog is not None:
            raise RubricError("Provide only one of skills=... or skills_catalog=..., not both.")

        if skills_catalog is None:
            if skills is None:
                skills_catalog = DEFAULT_SKILLS
            else:
                # Convert the list of SkillSpec into an ordered catalog
                tmp: Dict[str, SkillSpec] = {}
                for spec in skills:
                    if spec.skill_id in tmp:
                        raise RubricError(f"Duplicate skill_id in skills list: {spec.skill_id}")
                    tmp[spec.skill_id] = spec
                skills_catalog = tmp

        self.templates_dir: Path = _resolve_templates_dir(templates_dir)
        self.encoding = encoding
        self.system_prompt_path = self.templates_dir / system_prompt_filename
        self.output_schema_path = self.templates_dir / output_schema_filename

        # Load templates
        self._system_prompt_template: str = _read_text(self.system_prompt_path, encoding=self.encoding)
        self._base_output_schema_text: str = _read_text(self.output_schema_path, encoding=self.encoding)

        try:
            self._base_output_schema_obj: Dict[str, Any] = json.loads(self._base_output_schema_text)
        except Exception as e:
            raise RubricError(f"Invalid JSON in output schema template: {self.output_schema_path}") from e

        # Catalog + canonical order
        # (dict preserves insertion order in 3.7+, so DEFAULT_SKILLS order is canonical)
        self._skills_catalog: Dict[str, SkillSpec] = dict(skills_catalog)
        self._canonical_ids: List[str] = list(self._skills_catalog.keys())

        # Load skill markdown files
        self._skills_md: Dict[str, str] = {}
        for sid, spec in self._skills_catalog.items():
            skill_path = (self.templates_dir / spec.filename).expanduser().resolve()
            text = _read_text(skill_path, encoding=self.encoding)
            if validate_skill_markdown:
                _ensure_no_headings_leq_3(text, where=str(skill_path))
            self._skills_md[sid] = text

    # ---------------------------
    # Public helpers
    # ---------------------------

    def available_skills(self) -> List[Tuple[str, str]]:
        """Return [(skill_id, display_name), ...] in canonical order."""
        return [(sid, self._skills_catalog[sid].display_name) for sid in self._canonical_ids]

    def get_output_schema_text(self, skill_ids: Optional[Sequence[str]] = None) -> str:
        """Return the (optionally pruned) output schema template text."""
        return self.build_output_schema_text(skill_ids=skill_ids)

    def get_output_schema(self, skill_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Return the (optionally pruned) output schema template as a Python dict."""
        return self.build_output_schema(skill_ids=skill_ids)

    # ---------------------------
    # Core builders
    # ---------------------------

    def normalize_skill_ids(self, skill_ids: Optional[Sequence[str]]) -> List[str]:
        """
        Normalize and validate a skill selection, returning IDs in canonical order.
        - If skill_ids is None: all available skills in canonical order.
        - If provided: unknown IDs raise; duplicates are removed (stable).
        """
        if skill_ids is None:
            return list(self._canonical_ids)

        chosen_raw = list(skill_ids)
        if len(chosen_raw) == 0:
            raise RubricError("skill_ids was provided but empty. Provide at least one skill_id.")

        chosen = _dedupe_preserve_order(chosen_raw)

        unknown = [sid for sid in chosen if sid not in self._skills_catalog]
        if unknown:
            available = ", ".join(self._canonical_ids)
            raise RubricError(f"Unknown skill_ids: {unknown}. Available: {available}")

        chosen_set = set(chosen)
        return [sid for sid in self._canonical_ids if sid in chosen_set]

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
        Compile the selected skills into one Markdown block.

        Important: Skills are inserted under an existing "###" heading, so:
          header_level must be >= 4
        """
        if header_level < 4:
            raise ValueError("header_level must be >= 4 because skills are inserted under a '###' heading.")

        chosen_in_order = self.normalize_skill_ids(skill_ids)

        prefix = "#" * header_level
        blocks: List[str] = []

        for i, sid in enumerate(chosen_in_order, start=1):
            spec = self._skills_catalog[sid]
            title_parts: List[str] = []
            if numbered:
                title_parts.append(f"{i}.")
            title_parts.append(spec.display_name)
            if include_ids_in_heading:
                title_parts.append(f"(`{sid}`)")
            heading = f"{prefix} " + " ".join(title_parts)
            blocks.append(f"{heading}\n\n{self._skills_md[sid].rstrip()}")

        return separator.join(blocks).rstrip() + "\n"

    def build_output_schema(
        self,
        skill_ids: Optional[Sequence[str]] = None,
        *,
        validate_against_base: bool = True,
    ) -> Dict[str, Any]:
        """
        Build the output schema object shown to the model, pruning skill_results
        to contain *only* the selected skills.

        validate_against_base:
          - True: require that each selected skill exists in the base schema's
            "skill_results" keys (helps catch drift).
          - False: if missing, auto-insert "<pass|fail|uncertain>" placeholders.
        """
        chosen_in_order = self.normalize_skill_ids(skill_ids)
        base = deepcopy(self._base_output_schema_obj)

        # Ensure skill_results exists and is a dict
        sr = base.get("skill_results")
        if not isinstance(sr, dict):
            sr = {}
            base["skill_results"] = sr

        if validate_against_base:
            missing = [sid for sid in chosen_in_order if sid not in sr]
            if missing:
                raise RubricError(
                    "Selected skill_ids are missing from base output_schema.json skill_results keys: "
                    f"{missing}"
                )

        # Rebuild skill_results in canonical chosen order
        new_sr: Dict[str, Any] = {}
        for sid in chosen_in_order:
            if sid in sr:
                new_sr[sid] = sr[sid]
            else:
                new_sr[sid] = "<pass|fail|uncertain>"
        base["skill_results"] = new_sr
        return base

    def build_output_schema_text(
        self,
        skill_ids: Optional[Sequence[str]] = None,
        *,
        validate_against_base: bool = True,
    ) -> str:
        """Pretty JSON string for the pruned output schema."""
        return _pretty_json(self.build_output_schema(skill_ids=skill_ids, validate_against_base=validate_against_base))

    def _patch_skill_results_constraint(self, system_prompt: str, chosen_in_order: Sequence[str]) -> str:
        """
        Patch the constraint line about "skill_results" if the template contains it.

        This prevents the prompt from contradicting a pruned output_schema when
        fewer (or more) skills are assigned.
        """
        skills_list = ", ".join(f"`{sid}`" for sid in chosen_in_order)
        replacement = (
            f'4. `"skill_results"` MUST include results for all assigned skills '
            f'({len(chosen_in_order)}): {skills_list} with values `"pass"|"fail"|"uncertain"`.'
        )
        if not _SKILL_RESULTS_CONSTRAINT_RE.search(system_prompt):
            return system_prompt
        return _SKILL_RESULTS_CONSTRAINT_RE.sub(replacement, system_prompt, count=1)

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
        # Backwards compatible parameter (old code may pass it).
        # New behavior: validates only that selected skills exist in the base schema (subset),
        # and/or auto-fills placeholders if validate_schema_skills=False.
        validate_schema_skills: bool = True,
    ) -> str:
        """
        Build the full prompt by formatting system_prompt.md.

        evidence_pack:
          - str: inserted as-is
          - dict/list: pretty-printed JSON for readability
        """
        chosen_in_order = self.normalize_skill_ids(skill_ids)

        skills_block = self.build_skills_block(
            skill_ids=chosen_in_order,
            header_level=header_level,
        )

        output_schema_text = self.build_output_schema_text(
            skill_ids=chosen_in_order,
            validate_against_base=validate_schema_skills,
        )

        if isinstance(evidence_pack, str):
            evidence_text = evidence_pack.strip()
        else:
            evidence_text = _pretty_json(evidence_pack)

        if analysis_date is None:
            analysis_date = self._default_analysis_date(timezone)

        patched_system_prompt = self._patch_skill_results_constraint(
            self._system_prompt_template,
            chosen_in_order,
        )

        try:
            return patched_system_prompt.format(
                receipt_id=receipt_id,
                persona_name=persona_name,
                persona_description=persona_description,
                skills=skills_block,
                output_schema=output_schema_text,
                evidence_pack=evidence_text,
                analysis_date=analysis_date,  # optional placeholder
                timezone=timezone,            # optional placeholder
            ).rstrip() + "\n"
        except KeyError as e:
            raise RubricError(
                f"Missing placeholder in build_prompt() arguments: {e}. "
                "Check system_prompt.md placeholders."
            ) from e

    # ---------------------------
    # Internals
    # ---------------------------

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