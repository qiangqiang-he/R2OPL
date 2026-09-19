r"""Self-contained boxed-answer verification for Math, Science, and Logic.

All R2OPL-specific answer extraction and comparison logic lives in this file.
The only optional external dependency is ``math_verify``, used for symbolic
mathematics after deterministic choice, exact-text, and numeric checks.
"""

from __future__ import annotations

import ast
import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

try:
    from math_verify import parse as parse_math
    from math_verify import verify as verify_math
except ImportError:  # Choice, exact, and decimal checks remain available.
    parse_math = None
    verify_math = None


_BOXED_RE = re.compile(r"\\boxed\s*\{")
_CHOICE_LINE_RE = re.compile(r"(?m)^\s*([A-J])\s*[.)：:]\s+\S")
_STYLE_COMMAND_RE = re.compile(
    r"\\(?:text|textrm|textnormal|mathrm|mathbf|mathit|mathsf|mathtt|"
    r"textbf|operatorname|mbox)\s*\{"
)
_DECIMAL_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)


def _repair(text: object) -> str:
    return (
        str(text or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\ufe68", "\\")
        .replace("\uff3c", "\\")
        .replace("−", "-")
        .replace("–", "-")
        .replace("＋", "+")
    )


def _is_escaped(text: str, position: int) -> bool:
    backslashes = 0
    position -= 1
    while position >= 0 and text[position] == "\\":
        backslashes += 1
        position -= 1
    return bool(backslashes % 2)


def _matching_brace(text: str, opening: int) -> int | None:
    if opening >= len(text) or text[opening] != "{":
        return None
    depth = 1
    for position in range(opening + 1, len(text)):
        if _is_escaped(text, position):
            continue
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    return None


def extract_final_answer(response: str) -> tuple[str, bool]:
    r"""Extract the final balanced, non-empty ``\boxed{...}`` payload.

    The last box is authoritative.  If it is truncated, an earlier
    intermediate box is not silently substituted for the missing final answer.
    """

    response = _repair(response)
    matches = list(_BOXED_RE.finditer(response))
    if not matches:
        return "", False
    brace_start = matches[-1].end() - 1
    brace_end = _matching_brace(response, brace_start)
    if brace_end is None:
        return "", False
    answer = response[brace_start + 1 : brace_end].strip()
    return answer, bool(answer)


def extract_boxed_answers(response: str) -> list[str]:
    """Return every complete boxed payload in textual marker order."""

    response = _repair(response)
    answers: list[str] = []
    for match in _BOXED_RE.finditer(response):
        brace_start = match.end() - 1
        brace_end = _matching_brace(response, brace_start)
        if brace_end is not None:
            answers.append(response[brace_start + 1 : brace_end].strip())
    return answers


def _strip_surrounding_markup(text: object) -> str:
    value = _repair(text).strip()
    changed = True
    while value and changed:
        changed = False
        for left, right in (
            ("$$", "$$"),
            (r"\[", r"\]"),
            (r"\(", r"\)"),
            ("$", "$"),
            ("**", "**"),
            ("__", "__"),
            ("`", "`"),
        ):
            if value.startswith(left) and value.endswith(right):
                value = value[len(left) : len(value) - len(right)].strip()
                changed = True
                break
        if value.startswith("{") and _matching_brace(value, 0) == len(value) - 1:
            value = value[1:-1].strip()
            changed = True
    return value


def _replace_style_commands(text: str) -> str:
    """Remove LaTeX presentation commands while preserving their contents."""

    value = text
    while True:
        match = _STYLE_COMMAND_RE.search(value)
        if match is None:
            return value
        opening = match.end() - 1
        closing = _matching_brace(value, opening)
        if closing is None:
            return value
        value = value[: match.start()] + value[opening + 1 : closing] + value[closing + 1 :]


def extract_choice_label(answer: object) -> str:
    r"""Return an explicit A--J option label from a boxed-answer payload.

    Accepted forms include a bare label, harmless LaTeX/Markdown wrappers,
    ``A. option content``, and ``Choice A: option content``.  Content without
    an explicit option label intentionally returns an empty string.
    """

    value = _strip_surrounding_markup(answer)
    value = _replace_style_commands(value)
    value = _strip_surrounding_markup(value)
    value = value.replace("**", "").replace("__", "").replace("`", "")
    value = re.sub(r"^[\s#>*✅☑🎯]+", "", value).strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(
        r"^(?:(?:the\s+)?(?:final|correct)\s+)?(?:answer|choice|option)"
        r"\s*(?:is\s*|[:=\-]\s*)?",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    match = re.match(
        r"^[\[(]?\s*([A-J])\s*[\])]?\s*(?=$|[.:：\-]|\s)",
        value,
        flags=re.IGNORECASE,
    )
    if match is None:
        return ""
    # A bare lowercase letter is a harmless variant, but a lowercase algebraic
    # expression such as ``b - a`` must not be mistaken for an MCQ label plus
    # option content.
    if match.group(1).islower() and value[match.end() :].strip():
        return ""
    return match.group(1).upper()


def choice_labels(question: str) -> list[str]:
    """Return distinct printed option labels, or an empty list for open tasks."""

    labels: list[str] = []
    for match in _CHOICE_LINE_RE.finditer(_repair(question)):
        label = match.group(1).upper()
        if label not in labels:
            labels.append(label)
    return labels if len(labels) >= 2 else []


def inspect_answer_format(question: str, response: str) -> dict[str, object]:
    """Report strict box validity and explicit MCQ-label conformance."""

    labels = choice_labels(question)
    boxed_answers = extract_boxed_answers(response)
    final_answer, valid = extract_final_answer(response)
    normalized_choice_label = extract_choice_label(final_answer)
    multiple_choice = bool(labels)
    return {
        "valid": valid,
        "multiple_choice": multiple_choice,
        "choice_labels": labels,
        "box_count": len(boxed_answers),
        "boxed_answer": final_answer,
        "normalized_choice_label": normalized_choice_label,
        "choice_label_valid": (
            not multiple_choice or normalized_choice_label in labels
        ),
    }


def _normalize(text: object) -> str:
    value = _strip_surrounding_markup(text)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\displaystyle", "")
    value = _replace_style_commands(value)
    value = _strip_surrounding_markup(value)
    for spacing in (r"\,", r"\!", r"\;", r"\:", r"\quad", r"\qquad"):
        value = value.replace(spacing, "")
    value = re.sub(r"[，。.,;!?]+$", "", value.strip())
    return re.sub(r"\s+", "", value).casefold()


def _decimal_value(text: str) -> Decimal | None:
    if _DECIMAL_RE.fullmatch(text) is None:
        return None
    try:
        value = Decimal(text)
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def _math_payload(text: object) -> str:
    value = _strip_surrounding_markup(text)
    return value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")


def _replace_constant_latex(text: str) -> str | None:
    r"""Translate constant ``\frac`` and ``\sqrt`` constructs to Python syntax."""

    value = text
    construct_re = re.compile(r"\\(?:frac|sqrt)\b")
    while True:
        matches = list(construct_re.finditer(value))
        if not matches:
            return value
        match = matches[-1]
        command = match.group(0)[1:]
        cursor = match.end()
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1

        if command == "frac":
            if cursor >= len(value) or value[cursor] != "{":
                return None
            numerator_end = _matching_brace(value, cursor)
            if numerator_end is None:
                return None
            denominator_start = numerator_end + 1
            while denominator_start < len(value) and value[denominator_start].isspace():
                denominator_start += 1
            if denominator_start >= len(value) or value[denominator_start] != "{":
                return None
            denominator_end = _matching_brace(value, denominator_start)
            if denominator_end is None:
                return None
            numerator = value[cursor + 1 : numerator_end]
            denominator = value[denominator_start + 1 : denominator_end]
            replacement = f"(({numerator})/({denominator}))"
            value = value[: match.start()] + replacement + value[denominator_end + 1 :]
            continue

        if cursor >= len(value) or value[cursor] != "{":
            return None
        radicand_end = _matching_brace(value, cursor)
        if radicand_end is None:
            return None
        radicand = value[cursor + 1 : radicand_end]
        value = value[: match.start()] + f"sqrt({radicand})" + value[radicand_end + 1 :]


def _constant_math_expression(text: object) -> str | None:
    value = _math_payload(text)
    if "=" in value:
        value = value.rsplit("=", 1)[-1].strip()
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\displaystyle", "")
    value = value.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    value = value.replace(r"\pi", "pi")
    value = value.replace(r"^\circ", "").replace(r"\degree", "").replace("°", "")
    value = value.replace(r"\approx", "").replace(r"\sim", "")
    for spacing in (r"\,", r"\!", r"\;", r"\:", r"\quad", r"\qquad"):
        value = value.replace(spacing, "")
    converted = _replace_constant_latex(value)
    if converted is None:
        return None
    value = converted.replace("{", "(").replace("}", ")").replace("^", "**")
    value = value.replace("$", "").replace("&", "")
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"(\d|\))(?=(?:sqrt|pi|\())", r"\1*", value)
    value = re.sub(r"(pi|\))(?=(?:\d|\())", r"\1*", value)
    if not value or len(value) > 512 or "\\" in value:
        return None
    return value


def _evaluate_constant_node(node: ast.AST) -> Fraction | float:
    if isinstance(node, ast.Expression):
        return _evaluate_constant_node(node.body)
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
        if isinstance(node.value, int):
            return Fraction(node.value)
        if isinstance(node.value, float) and math.isfinite(node.value):
            return Fraction(str(node.value))
        raise ValueError("unsupported constant")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_constant_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_constant_node(node.left)
        right = _evaluate_constant_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            if not isinstance(right, Fraction) or right.denominator != 1:
                raise ValueError("non-integer power")
            exponent = right.numerator
            if abs(exponent) > 1000:
                raise ValueError("exponent too large")
            return left**exponent
        raise ValueError("unsupported operator")
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sqrt"
        and len(node.args) == 1
        and not node.keywords
    ):
        argument = _evaluate_constant_node(node.args[0])
        if argument < 0:
            raise ValueError("negative square root")
        if isinstance(argument, Fraction):
            numerator = math.isqrt(argument.numerator)
            denominator = math.isqrt(argument.denominator)
            if (
                numerator * numerator == argument.numerator
                and denominator * denominator == argument.denominator
            ):
                return Fraction(numerator, denominator)
        return math.sqrt(float(argument))
    raise ValueError("unsupported expression")


def _constant_math_value(text: object) -> Fraction | float | None:
    expression = _constant_math_expression(text)
    if expression is None:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            return None
        value = _evaluate_constant_node(tree)
        return value if math.isfinite(float(value)) else None
    except (ArithmeticError, SyntaxError, TypeError, ValueError, OverflowError):
        return None


def _constant_math_equal(prediction: object, target: object) -> bool:
    left = _constant_math_value(prediction)
    right = _constant_math_value(target)
    if left is None or right is None:
        return False
    if isinstance(left, Fraction) and isinstance(right, Fraction):
        return left == right
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)


def _symbolic_equal(prediction: str, target: str) -> bool:
    if parse_math is None or verify_math is None:
        return False
    try:
        parsed_prediction = parse_math(f"${_math_payload(prediction)}$")
        parsed_target = parse_math(f"${_math_payload(target)}$")
        return bool(
            parsed_prediction
            and parsed_target
            and verify_math(parsed_target, parsed_prediction)
        )
    except Exception:
        return False


def answers_equivalent(prediction: str, target: str) -> bool:
    """Compare a boxed payload with its gold answer.

    A gold answer that is an explicit option label activates strict MCQ mode:
    the prediction must also contain an explicit label.  Option text may follow
    that label, but option text by itself is never inferred or accepted.
    """

    target_normalized = _normalize(target)
    target_label = extract_choice_label(target)
    if target_label and target_normalized.casefold() == target_label.casefold():
        prediction_label = extract_choice_label(prediction)
        return bool(prediction_label) and prediction_label == target_label

    prediction_normalized = _normalize(prediction)
    if not prediction_normalized or not target_normalized:
        return False
    if prediction_normalized == target_normalized:
        return True

    prediction_number = _decimal_value(prediction_normalized)
    target_number = _decimal_value(target_normalized)
    if prediction_number is not None and target_number is not None:
        return prediction_number == target_number
    if _constant_math_equal(prediction, target):
        return True
    return _symbolic_equal(prediction, target)


def verify_response_answer(response: str, answer: str) -> float:
    """Return 1.0 only for a complete boxed answer equivalent to ``answer``."""

    prediction, valid_format = extract_final_answer(response)
    if not valid_format:
        return 0.0
    return float(answers_equivalent(prediction, str(answer)))


__all__ = [
    "answers_equivalent",
    "choice_labels",
    "extract_choice_label",
    "extract_boxed_answers",
    "extract_final_answer",
    "inspect_answer_format",
    "verify_response_answer",
]
