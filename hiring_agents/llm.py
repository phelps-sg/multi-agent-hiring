"""LLM client implementations.

Provides both a simulated client (for testing) and a real client (via LiteLLM)
that can call any supported model. The LLMClient protocol allows the agents
to be tested deterministically while supporting real LLM calls in production.
"""

from __future__ import annotations

import json
import typing as t

from hiring_agents.sub_agents.summariser import LLMClient


class SimulatedLLMClient:
    """Deterministic LLM for testing — returns predictable outputs."""

    def summarise(self, text: str, focus: str) -> tuple[str, list[str]]:
        word_count = len(text.split())
        summary = (
            f"Application summary (focus: {focus}). "
            f"The candidate submitted {word_count} words of content."
        )
        key_points = [
            f"Focus area: {focus}",
            f"Content length: {word_count} words",
        ]
        return summary, key_points

    def synthesise_reasoning(
        self,
        criteria_summary: str,
        application_summary: str,
        match: str,
        action: str,
    ) -> str:
        return (
            f"Deterministic reasoning: {criteria_summary} "
            f"Match: {match}. Recommendation: {action}."
        )


class LiteLLMClient:
    """Real LLM client via LiteLLM — calls an actual model."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        monitoring_context: str | None = None,
    ) -> None:
        self.model = model
        self.monitoring_context = monitoring_context

    def _call(self, messages: list[dict[str, str]]) -> str:
        import litellm

        response = litellm.completion(model=self.model, messages=messages)
        return response.choices[0].message.content

    def summarise(self, text: str, focus: str) -> tuple[str, list[str]]:
        system_content = (
            "You are a document summarisation agent in a hiring pipeline. "
            "You produce concise, factual summaries of candidate application documents. "
            "You do not evaluate or recommend — you only summarise.\n\n"
            f"Focus your summary on aspects relevant to: {focus}\n\n"
            "Respond in JSON with two fields:\n"
            '  "summary": a 2-3 sentence summary\n'
            '  "key_points": a list of 3-5 key factual points\n\n'
            "Do not include any other text outside the JSON."
        )
        if self.monitoring_context:
            system_content += f"\n\n{self.monitoring_context}"
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": f"Summarise the following application documents:\n\n{text}",
            },
        ]
        raw = self._call(messages)
        try:
            # Strip markdown code fences if present (e.g. ```json ... ```)
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]  # drop opening fence line
                cleaned = cleaned.rsplit("```", 1)[0]  # drop closing fence
            parsed = json.loads(cleaned)
            return parsed["summary"], parsed["key_points"]
        except (json.JSONDecodeError, KeyError):
            # Fallback: treat the whole response as the summary
            return raw, []

    def synthesise_reasoning(
        self,
        criteria_summary: str,
        application_summary: str,
        match: str,
        action: str,
    ) -> str:
        """Produce a natural-language reasoning narrative for the screening decision.

        The match level and action have already been determined deterministically.
        The LLM's role is to produce a readable explanation — it cannot change
        the decision.
        """
        system_content = (
            "You are a screening agent producing a reasoning narrative for a "
            "hiring decision that has already been made by deterministic rules. "
            "Your role is to explain the decision clearly and concisely — "
            "you cannot change the match level or recommended action.\n\n"
            "Be factual and direct. Reference specific criteria results. "
            "2-4 sentences maximum."
        )
        if self.monitoring_context:
            system_content += f"\n\n{self.monitoring_context}"
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": (
                    f"The deterministic evaluation produced:\n"
                    f"  Match: {match}\n"
                    f"  Recommended action: {action}\n\n"
                    f"Criteria results:\n{criteria_summary}\n\n"
                    f"Application summary:\n{application_summary}\n\n"
                    f"Write a concise reasoning narrative explaining this decision."
                ),
            },
        ]
        return self._call(messages)
