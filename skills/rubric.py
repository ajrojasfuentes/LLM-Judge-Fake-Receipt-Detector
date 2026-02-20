"""
Rubric: compiles the 5 forensic skills into a structured judge prompt.
"""

from pathlib import Path
import json

TEMPLATES_DIR = Path(__file__).parent / "templates"

SKILL_FILES = {
    "math_consistency": "math_consistency.txt",
    "typography_analysis": "typography_analysis.txt",
    "visual_authenticity": "visual_authenticity.txt",
    "layout_structure": "layout_structure.txt",
    "contextual_validation": "contextual_validation.txt",
}


class Rubric:
    """
    Loads and compiles skill templates into a complete judge prompt.

    Usage:
        rubric = Rubric()
        prompt = rubric.build_prompt(
            receipt_id="receipt_001",
            persona_name="Forensic Accountant",
            persona_description="You are a ...",
            focus_skills=["math_consistency", "contextual_validation"],  # optional filter
        )
    """

    def __init__(self):
        self._skills: dict[str, str] = {}
        self._system_prompt_template: str = ""
        self._output_schema: str = ""
        self._load_templates()

    def _load_templates(self) -> None:
        for skill_key, filename in SKILL_FILES.items():
            path = TEMPLATES_DIR / filename
            self._skills[skill_key] = path.read_text(encoding="utf-8").strip()

        self._system_prompt_template = (
            TEMPLATES_DIR / "system_prompt.txt"
        ).read_text(encoding="utf-8").strip()

        self._output_schema = (
            TEMPLATES_DIR / "output_schema.json"
        ).read_text(encoding="utf-8").strip()

    def build_prompt(
        self,
        receipt_id: str,
        persona_name: str,
        persona_description: str,
        focus_skills: list[str] | None = None,
    ) -> str:
        """
        Build the complete system prompt for a judge.

        Args:
            receipt_id: Identifier for the receipt being analyzed.
            persona_name: Display name of the judge persona.
            persona_description: Role description injected into the prompt.
            focus_skills: Optional subset of skills to emphasize (all are always included,
                          but focus_skills get an extra emphasis marker).
        Returns:
            The complete formatted prompt string.
        """
        focus_skills = focus_skills or list(SKILL_FILES.keys())

        def _maybe_emphasize(skill_key: str, content: str) -> str:
            if skill_key in focus_skills:
                return f">>> PRIMARY FOCUS <<<\n{content}"
            return content

        prompt = self._system_prompt_template.format(
            receipt_id=receipt_id,
            persona_name=persona_name,
            persona_description=persona_description,
            skill_math_consistency=_maybe_emphasize(
                "math_consistency", self._skills["math_consistency"]
            ),
            skill_typography_analysis=_maybe_emphasize(
                "typography_analysis", self._skills["typography_analysis"]
            ),
            skill_visual_authenticity=_maybe_emphasize(
                "visual_authenticity", self._skills["visual_authenticity"]
            ),
            skill_layout_structure=_maybe_emphasize(
                "layout_structure", self._skills["layout_structure"]
            ),
            skill_contextual_validation=_maybe_emphasize(
                "contextual_validation", self._skills["contextual_validation"]
            ),
            output_schema=self._output_schema,
        )
        return prompt

    def get_output_schema(self) -> dict:
        """Return the output schema as a Python dict."""
        return json.loads(self._output_schema)
