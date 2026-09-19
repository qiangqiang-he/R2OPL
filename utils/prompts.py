"""Cross-model prompts for PG-OPD.

The task instruction is intentionally kept separate from model chat templates.
Qwen3 and Gemini 4 can therefore use the same task contract while each
tokenizer still supplies its own control tokens and generation prefix.  The
``gemma`` aliases are retained because local/Hugging Face Gemini-family
checkpoints may use Gemma model identifiers.
"""

from __future__ import annotations

from typing import Any


CROSS_DOMAIN_PROMPT = r"""Solve the problem step by step. Put the final answer in `\boxed{{...}}`.

For a multiple-choice problem, put the option label (for example, `\boxed{{B}}`) in the box rather than the option text. For a non-multiple-choice problem, put the final answer itself in the box.

### Problem
{question}"""

EXPLICIT_STEP_PROMPT = r"""Solve the problem step by step. Organize the reasoning with headings `### Step 1`, `### Step 2`, and so on. Put the final answer in `\boxed{{...}}`.

For a multiple-choice problem, put the option label (for example, `\boxed{{B}}`) in the box rather than the option text. For a non-multiple-choice problem, put the final answer itself in the box.

### Problem
{question}"""

# Stable public names for configuration files and experiment manifests.  The
# explicit-step template is the project-wide default; callers can still opt in
# to the less structured cross-domain template by name.
CROSS_DOMAIN_PROMPT_NAME = "cross_domain_prompt"
EXPLICIT_STEP_PROMPT_NAME = "explicit_step_prompt"
PROMPT_NAME = EXPLICIT_STEP_PROMPT_NAME
PROMPT_TEMPLATES = {
    CROSS_DOMAIN_PROMPT_NAME: CROSS_DOMAIN_PROMPT,
    EXPLICIT_STEP_PROMPT_NAME: EXPLICIT_STEP_PROMPT,
}
SUPPORTED_MODEL_FAMILIES = ("qwen3", "gemini4")


def _validated_question(question: str) -> str:
    question = str(question).strip()
    if not question:
        raise ValueError("The reasoning prompt requires a non-empty question.")
    return question


def render_prompt(*, question: str) -> str:
    """Render the project-default model-independent task prompt."""

    return PROMPT_TEMPLATES[PROMPT_NAME].format(
        question=_validated_question(question)
    )


def render_explicit_step_prompt(*, question: str) -> str:
    """Render the optional prompt that requests explicit step headings."""

    return EXPLICIT_STEP_PROMPT.format(question=_validated_question(question))


def build_prompt_messages(
    *,
    question: str,
    prompt_name: str = PROMPT_NAME,
) -> list[dict[str, str]]:
    """Return chat-template input accepted by both Qwen3 and Gemma.

    A single user message is deliberate: some Gemma instruction templates do
    not accept a separate system role, while both model families accept user
    instructions. No model-specific control tokens are embedded here.
    """

    try:
        template = PROMPT_TEMPLATES[prompt_name]
    except KeyError as error:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown prompt {prompt_name!r}; expected one of: {available}."
        ) from error
    content = template.format(question=_validated_question(question))
    return [{"role": "user", "content": content}]


def normalize_model_family(model_family: str) -> str:
    """Normalize a family name or model identifier to a supported family."""

    normalized = str(model_family).strip().lower()
    if "qwen3" in normalized:
        return "qwen3"
    if any(
        alias in normalized
        for alias in ("gemini4", "gemini-4", "gemma4", "gemma-4", "gemma")
    ):
        return "gemini4"
    supported = ", ".join(SUPPORTED_MODEL_FAMILIES)
    raise ValueError(
        f"Unsupported model family {model_family!r}; expected one of: {supported}."
    )


def render_chat_prompt(
    tokenizer: Any,
    *,
    question: str,
    model_family: str,
    prompt_name: str = PROMPT_NAME,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
) -> Any:
    """Apply the project's fixed native chat-template behavior.

    The returned value follows ``tokenizer.apply_chat_template``: normally a
    string when ``tokenize=False`` and token IDs otherwise.
    """

    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("The tokenizer must provide apply_chat_template().")

    family = normalize_model_family(model_family)
    template_kwargs: dict[str, Any] = {
        "tokenize": bool(tokenize),
        "add_generation_prompt": bool(add_generation_prompt),
        "enable_thinking": False,
    }

    return tokenizer.apply_chat_template(
        build_prompt_messages(question=question, prompt_name=prompt_name),
        **template_kwargs,
    )


__all__ = [
    "CROSS_DOMAIN_PROMPT",
    "CROSS_DOMAIN_PROMPT_NAME",
    "EXPLICIT_STEP_PROMPT",
    "EXPLICIT_STEP_PROMPT_NAME",
    "PROMPT_NAME",
    "PROMPT_TEMPLATES",
    "SUPPORTED_MODEL_FAMILIES",
    "build_prompt_messages",
    "normalize_model_family",
    "render_chat_prompt",
    "render_explicit_step_prompt",
    "render_prompt",
]
