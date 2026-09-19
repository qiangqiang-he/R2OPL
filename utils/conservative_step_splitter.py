"""Conservative, token-aligned step splitting for reasoning rollouts.

The splitter deliberately prefers merging over a speculative boundary.  Text
is used only to discover semantic boundaries.  Token spans are recovered from
prefix decodes of the original rollout IDs; step text is never re-tokenized.

This module is standalone and depends only on the Python standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class TextStep:
    """One source-preserving text step."""

    index: int
    char_start: int
    char_end: int
    text: str
    kind: str
    boundary_reason: str
    confidence: str


@dataclass(frozen=True)
class TokenStep:
    """One text step aligned to a contiguous span of original rollout IDs."""

    index: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    text: str
    kind: str
    boundary_reason: str
    confidence: str


@dataclass(frozen=True)
class _Line:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _Marker:
    position: int
    family: str
    number: int
    heading_level: int
    reason: str
    confidence: str


@dataclass(frozen=True)
class _Boundary:
    position: int
    reason: str
    confidence: str


_EXPLICIT_STEP_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:(?P<heading>#{1,6})[ \t]+)?"
    r"(?:[✅☑🎯]\ufe0f?[ \t]*)?(?:\*{1,2}[ \t]*)?"
    r"(?P<family>solution[ \t]+step|step|part|case|operation|phase|stage)"
    r"[ \t]*(?:no\.?[ \t]*)?#?[ \t]*"
    r"(?P<number>\d+|[ivxlcdm]+)\b",
    re.IGNORECASE,
)
_BOLD_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:(?P<heading>#{1,6})[ \t]+)?"
    r"\*\*(?P<number>\d+)[.)][ \t]+.+\*\*[ \t]*(?:[.:：][ \t]*)?$"
)
_MARKDOWN_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<heading>#{1,6})[ \t]+"
    r"(?:\*{1,2})?(?P<number>\d+)[.)][ \t]+\S.+$"
)
_NUMBERED_BOLD_LABEL_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<number>\d+)[.)][ \t]+"
    r"\*{1,2}(?P<title>[^*\r\n]{2,180})\*{1,2}(?P<tail>[^\r\n]*)$"
)
_MARKDOWN_HEADING_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<heading>#{1,6})[ \t]+(?P<title>\S.*?)[ \t]*$"
)
_BOLD_HEADING_RE = re.compile(
    r"^(?P<indent>[ \t]*)\*{2}(?P<title>[^*\r\n]{2,180})\*{2}"
    r"[ \t]*(?:[.:：][ \t]*)?$"
)
_OPTION_HEADING_RE = re.compile(
    r"^(?:(?:option|choice)[ \t]+)?[A-J](?:[.)\]:：-]|[ \t]*$)",
    re.IGNORECASE,
)
_NON_REASONING_HEADING_RE = re.compile(
    r"^(?:question|choices?|options?|answer[ \t]+choices?|problem(?:[ \t]+statement)?|"
    r"given(?:[ \t]+conditions?)?|passage|goal|background(?:[ \t]+theory)?|"
    r"key[ \t]+(?:points?|information|concepts?|elements?|properties)"
    r"(?:[ \t]+(?:from|of)[ \t]+(?:the[ \t]+)?(?:problem|passage|argument))?|"
    r"definitions?|definition[ \t]+recap|passage[ \t]+summary|"
    r"argument[ \t]+summary|summary[ \t]+of[ \t]+the[ \t]+"
    r"(?:argument|passage|views?)|the[ \t]+issue)$",
    re.IGNORECASE,
)
_REASONING_LABEL_RE = re.compile(
    r"\b(?:step|premise|assumption|observation|inference|deduction|analysis|"
    r"case|calculat(?:e|ion)|deriv(?:e|ation)|determine|evaluat(?:e|ion)|"
    r"eliminate|conclusion|result|substitut(?:e|ion)|simplif(?:y|ication)|"
    r"comput(?:e|ation)|setup|identify|interpret|compare|apply|verify|check|"
    r"solve|proof|approach|method)\b",
    re.IGNORECASE,
)
_STRUCTURED_COMPONENT_HEADING_RE = re.compile(
    r"^(?:(?:first|second|third|fourth|fifth|sixth|next|last|overall)[ \t]+"
    r"(?:term|component|expression|equation|factor|relation(?:ship)?|case|part)|"
    r"relation(?:ship)?[ \t]+between|hypothesis(?:[ \t]+based[ \t]+on)?|"
    r"assumption\b)",
    re.IGNORECASE,
)
_FINAL_HEADING_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+)?(?:[✅☑🎯]\ufe0f?[ \t]*)?"
    r"(?:\*{1,2}[ \t]*)?"
    r"(?:final[ \t]+(?:answer|result|conclusion)|correct[ \t]+answer|"
    r"最终答案|最终结果|结论)\b",
    re.IGNORECASE,
)
_FINAL_PARAGRAPH_CUE_RE = re.compile(
    r"^[ \t]*(?:\*{1,2}[ \t]*)?"
    r"(?:therefore|thus|hence|consequently|in[ \t]+conclusion|finally|"
    r"overall|we[ \t]+conclude|this[ \t]+gives|this[ \t]+yields|"
    r"the[ \t]+(?:final|correct)[ \t]+answer|the[ \t]+answer[ \t]+is|"
    r"answer[ \t]*:)",
    re.IGNORECASE,
)
_LIST_LINE_RE = re.compile(
    r"^[ \t]{0,3}(?:[-+*]|\d+[.)]|[A-Za-z][.)])[ \t]+\S"
)
_TABLE_SEPARATOR_RE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$"
)
_LATEX_BEGIN_RE = re.compile(
    r"\\begin\{(?P<env>equation\*?|align\*?|aligned|gather\*?|multline\*?|"
    r"cases|array|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|tabular)\}"
)
_BOX_PREFIX = r"\boxed{"


def _lines(text: str) -> list[_Line]:
    result: list[_Line] = []
    cursor = 0
    for value in text.splitlines(keepends=True):
        result.append(_Line(cursor, cursor + len(value), value))
        cursor += len(value)
    if cursor < len(text):
        result.append(_Line(cursor, len(text), text[cursor:]))
    return result


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if start >= end:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _delimiter_ranges(
    text: str,
    opening: re.Pattern[str],
    closing_factory: Any,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        begin = opening.search(text, cursor)
        if begin is None:
            return ranges
        closing = closing_factory(begin)
        end = closing.search(text, begin.end())
        if end is None:
            ranges.append((begin.start(), len(text)))
            return ranges
        ranges.append((begin.start(), end.end()))
        cursor = end.end()


def _fenced_code_ranges(text: str) -> list[tuple[int, int]]:
    opening = re.compile(r"(?m)^[ \t]{0,3}(?P<mark>`{3,}|~{3,})[^\n]*(?:\n|$)")

    def closing_factory(match: re.Match[str]) -> re.Pattern[str]:
        marker = match.group("mark")
        return re.compile(
            rf"(?m)^[ \t]{{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*(?:\n|$)"
        )

    return _delimiter_ranges(text, opening, closing_factory)


def _simple_delimiter_ranges(
    text: str,
    opening_pattern: str,
    closing_pattern: str,
    *,
    flags: int = 0,
) -> list[tuple[int, int]]:
    opening = re.compile(opening_pattern, flags)
    closing = re.compile(closing_pattern, flags)
    return _delimiter_ranges(text, opening, lambda _match: closing)


def _latex_environment_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        begin = _LATEX_BEGIN_RE.search(text, cursor)
        if begin is None:
            return ranges
        environment = begin.group("env")
        token_re = re.compile(
            rf"\\(?:begin|end)\{{{re.escape(environment)}\}}"
        )
        depth = 1
        end_position = len(text)
        for token in token_re.finditer(text, begin.end()):
            if token.group(0).startswith(r"\begin"):
                depth += 1
            else:
                depth -= 1
            if depth == 0:
                end_position = token.end()
                break
        ranges.append((begin.start(), end_position))
        if end_position == len(text):
            return ranges
        cursor = end_position


def _table_ranges(lines: list[_Line]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        run_start = index
        while index < len(lines):
            stripped = lines[index].text.strip()
            if not stripped or "|" not in stripped:
                break
            index += 1
        run = lines[run_start:index]
        if len(run) >= 2 and (
            any(_TABLE_SEPARATOR_RE.match(line.text) for line in run)
            or all(line.text.strip().startswith("|") for line in run)
        ):
            ranges.append((run[0].start, run[-1].end))
        if index == run_start:
            index += 1
    return ranges


def _continuous_list_ranges(lines: list[_Line]) -> list[tuple[int, int]]:
    """Protect compact lists from every boundary detector."""

    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        if _LIST_LINE_RE.match(lines[index].text) is None:
            index += 1
            continue
        run_start = index
        marker_count = 0
        while index < len(lines) and lines[index].text.strip():
            if _LIST_LINE_RE.match(lines[index].text):
                marker_count += 1
            elif not lines[index].text.startswith((" ", "\t")):
                break
            index += 1
        if marker_count >= 2:
            ranges.append((lines[run_start].start, lines[index - 1].end))
        if index == run_start:
            index += 1
    return ranges


def _protected_ranges(text: str, lines: list[_Line]) -> list[tuple[int, int]]:
    ranges = _fenced_code_ranges(text)
    ranges.extend(
        _simple_delimiter_ranges(text, r"(?<!\\)\$\$", r"(?<!\\)\$\$")
    )
    ranges.extend(_simple_delimiter_ranges(text, r"\\\[", r"\\\]"))
    ranges.extend(_simple_delimiter_ranges(text, r"<think\b[^>]*>", r"</think\s*>", flags=re.I))
    ranges.extend(_latex_environment_ranges(text))
    ranges.extend(_table_ranges(lines))
    ranges.extend(_continuous_list_ranges(lines))
    return _merge_ranges(ranges)


def _is_protected(position: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in ranges)


def _roman_to_int(value: str) -> int | None:
    value = value.upper()
    if not value or re.fullmatch(r"[IVXLCDM]+", value) is None:
        return None
    numbers = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    result = 0
    previous = 0
    for character in reversed(value):
        number = numbers[character]
        result += -number if number < previous else number
        previous = max(previous, number)
    return result


def _marker_number(value: str) -> int | None:
    return int(value) if value.isdigit() else _roman_to_int(value)


def _substantive_length(value: str) -> int:
    value = re.sub(r"```.*?```", "", value, flags=re.DOTALL)
    value = re.sub(r"[#*_`$\\{}\[\]()<>|:=+\-]", "", value)
    return len(re.sub(r"\s+", "", value))


def _ends_complete(value: str) -> bool:
    stripped = value.rstrip()
    if not stripped:
        return False
    if stripped.endswith((":", "：", ",", "，", ";", "；", "=", "\\")):
        return False
    without_display = re.sub(r"(?<!\\)\$\$", "", stripped)
    if len(re.findall(r"(?<!\\)\$", without_display)) % 2:
        return False
    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    return all(stripped.count(left) <= stripped.count(right) for left, right in pairs)


def _first_marker_boundary_allowed(text: str, position: int, minimum: int) -> bool:
    prefix = text[:position]
    return _substantive_length(prefix) >= minimum and _ends_complete(prefix)


def _increasing_runs(markers: list[_Marker]) -> list[list[_Marker]]:
    if not markers:
        return []
    runs: list[list[_Marker]] = []
    current = [markers[0]]
    for marker in markers[1:]:
        if marker.number > current[-1].number:
            current.append(marker)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = [marker]
    if len(current) >= 2:
        runs.append(current)
    return runs


def _ends_semantic_unit(value: str) -> bool:
    stripped = value.rstrip()
    if not _ends_complete(stripped):
        return False
    boxes = _balanced_boxes(stripped)
    if boxes and boxes[-1][1] == len(stripped):
        return True
    return bool(
        re.search(
            r"(?:[.!?。！？]|(?<!\\)\$|\\\]|\\\)|"
            r"\\end\{[^{}]+\}|```|~~~)\s*$",
            stripped,
        )
    )


def _complete_run_prefix(
    text: str,
    run: list[_Marker],
    minimum: int,
) -> list[_Marker]:
    """Keep only the complete prefix of a numbered marker run.

    A later sibling is evidence that the preceding member was completed.  The
    final member has no such evidence, so it also needs a syntactically closed
    ending.  An unfinished generated tail is thereby attached to the previous
    accepted step instead of becoming an incomplete step of its own.
    """

    complete: list[_Marker] = []
    for index, marker in enumerate(run):
        end = run[index + 1].position if index + 1 < len(run) else len(text)
        segment = text[marker.position:end]
        if _substantive_length(segment) < minimum:
            break
        if index + 1 == len(run) and not _ends_semantic_unit(segment):
            break
        complete.append(marker)
    return complete


def _explicit_boundaries(
    text: str,
    lines: list[_Line],
    protected: list[tuple[int, int]],
) -> list[_Boundary]:
    markers: list[_Marker] = []
    for line in lines:
        if _is_protected(line.start, protected):
            continue
        match = _EXPLICIT_STEP_RE.match(line.text)
        if match is None or match.group("indent"):
            continue
        number = _marker_number(match.group("number"))
        if number is None:
            continue
        family = match.group("family").lower().replace("solution ", "")
        heading = match.group("heading") or ""
        if family == "case" and not heading:
            continue
        markers.append(
            _Marker(
                position=line.start,
                family=family,
                number=number,
                heading_level=len(heading),
                reason="explicit_step_sequence",
                confidence="hard",
            )
        )

    valid_runs: list[list[_Marker]] = []
    signatures = sorted({(item.family, item.heading_level) for item in markers})
    for family, heading_level in signatures:
        siblings = [
            item
            for item in markers
            if item.family == family and item.heading_level == heading_level
        ]
        for run in _increasing_runs(siblings):
            complete = _complete_run_prefix(text, run, minimum=28)
            if len(complete) >= 2:
                valid_runs.append(complete)
    if not valid_runs:
        return []

    shallowest = min(run[0].heading_level for run in valid_runs)
    boundaries: list[_Boundary] = []
    for run in valid_runs:
        if run[0].heading_level != shallowest:
            continue
        for index, marker in enumerate(run):
            if index == 0 and not _first_marker_boundary_allowed(
                text, marker.position, minimum=80
            ):
                continue
            boundaries.append(
                _Boundary(marker.position, marker.reason, marker.confidence)
            )
    return boundaries


def _bold_numbered_heading_boundaries(
    text: str,
    lines: list[_Line],
    protected: list[tuple[int, int]],
) -> list[_Boundary]:
    """Split Gemma-style authored sections such as ``**1. Analyze ...**``.

    Requiring a complete bold heading, a consecutive sibling sequence, and
    substantive section bodies distinguishes these sections from ordinary
    numbered facts and multiple-choice options.
    """

    markers: list[_Marker] = []
    for line in lines:
        if _is_protected(line.start, protected):
            continue
        match = _BOLD_NUMBERED_HEADING_RE.match(line.text.rstrip("\r\n"))
        if match is None or match.group("indent"):
            continue
        markers.append(
            _Marker(
                position=line.start,
                family="bold_numbered_heading",
                number=int(match.group("number")),
                heading_level=len(match.group("heading") or ""),
                reason="bold_numbered_heading_sequence",
                confidence="hard",
            )
        )

    boundaries: list[_Boundary] = []
    signatures = sorted({marker.heading_level for marker in markers})
    for heading_level in signatures:
        siblings = [
            marker for marker in markers if marker.heading_level == heading_level
        ]
        for run in _increasing_runs(siblings):
            if any(
                right.number != left.number + 1
                for left, right in zip(run, run[1:])
            ):
                continue
            run = _complete_run_prefix(text, run, minimum=80)
            if len(run) < 2:
                continue
            for index, marker in enumerate(run):
                if index == 0 and not _first_marker_boundary_allowed(
                    text, marker.position, minimum=80
                ):
                    continue
                boundaries.append(
                    _Boundary(marker.position, marker.reason, marker.confidence)
                )
    return boundaries


def _markdown_numbered_heading_boundaries(
    text: str,
    lines: list[_Line],
    protected: list[tuple[int, int]],
) -> list[_Boundary]:
    """Split consecutive Markdown sections such as ``### 1. Derive ...``."""

    markers: list[_Marker] = []
    for line in lines:
        if _is_protected(line.start, protected):
            continue
        match = _MARKDOWN_NUMBERED_HEADING_RE.match(line.text.rstrip("\r\n"))
        if match is None or match.group("indent"):
            continue
        markers.append(
            _Marker(
                position=line.start,
                family="markdown_numbered_heading",
                number=int(match.group("number")),
                heading_level=len(match.group("heading")),
                reason="markdown_numbered_heading_sequence",
                confidence="hard",
            )
        )

    valid_runs: list[list[_Marker]] = []
    for heading_level in sorted({marker.heading_level for marker in markers}):
        siblings = [
            marker for marker in markers if marker.heading_level == heading_level
        ]
        for run in _increasing_runs(siblings):
            if any(
                right.number != left.number + 1
                for left, right in zip(run, run[1:])
            ):
                continue
            complete = _complete_run_prefix(text, run, minimum=60)
            if len(complete) >= 2:
                valid_runs.append(complete)
    if not valid_runs:
        return []

    shallowest = min(run[0].heading_level for run in valid_runs)
    boundaries: list[_Boundary] = []
    for run in valid_runs:
        if run[0].heading_level != shallowest:
            continue
        for index, marker in enumerate(run):
            if index == 0 and not _first_marker_boundary_allowed(
                text, marker.position, minimum=80
            ):
                continue
            boundaries.append(
                _Boundary(marker.position, marker.reason, marker.confidence)
            )
    return boundaries


def _numbered_reasoning_boundaries(
    text: str,
    lines: list[_Line],
    protected: list[tuple[int, int]],
) -> list[_Boundary]:
    """Split process-labelled numbered paragraphs, not ordinary fact lists."""

    markers: list[_Marker] = []
    labelled_positions: set[int] = set()
    for line in lines:
        if _is_protected(line.start, protected):
            continue
        match = _NUMBERED_BOLD_LABEL_RE.match(line.text.rstrip("\r\n"))
        if match is None or match.group("indent"):
            continue
        marker = _Marker(
            position=line.start,
            family="numbered_reasoning",
            number=int(match.group("number")),
            heading_level=0,
            reason="numbered_reasoning_sequence",
            confidence="hard",
        )
        markers.append(marker)
        if _REASONING_LABEL_RE.search(match.group("title")):
            labelled_positions.add(marker.position)

    boundaries: list[_Boundary] = []
    for run in _increasing_runs(markers):
        if any(
            right.number != left.number + 1
            for left, right in zip(run, run[1:])
        ):
            continue
        run = _complete_run_prefix(text, run, minimum=80)
        if len(run) < 2:
            continue
        labelled = sum(marker.position in labelled_positions for marker in run)
        if labelled < 2 or labelled * 2 < len(run):
            continue
        for index, marker in enumerate(run):
            if index == 0 and not _first_marker_boundary_allowed(
                text, marker.position, minimum=80
            ):
                continue
            boundaries.append(
                _Boundary(marker.position, marker.reason, marker.confidence)
            )
    return boundaries


def _normalized_heading_title(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^(?:\*{1,2}|_{1,2})[ \t]*", "", value)
    value = re.sub(
        r"[ \t]*(?:\*{1,2}|_{1,2})(?=[ \t]*(?:[.:：])?[ \t]*$)",
        "",
        value,
    )
    return value.strip().rstrip(".:：").strip()


def _is_semantic_heading(title: str, *, require_process_label: bool) -> bool:
    raw_title = title.strip()
    title = _normalized_heading_title(raw_title)
    if not title or len(title) > 120:
        return False
    if title.startswith(('"', "'", "“", "‘", "→", "⇒", "=>")):
        return False
    if _OPTION_HEADING_RE.match(title):
        return False
    if _NON_REASONING_HEADING_RE.fullmatch(title):
        return False
    if re.match(r"^\d+[.)][ \t]+", title):
        return False
    if _EXPLICIT_STEP_RE.match(title) or _FINAL_HEADING_RE.match(title):
        return False
    if _substantive_length(title) < 3:
        return False
    if require_process_label and not (
        _REASONING_LABEL_RE.search(title)
        or _STRUCTURED_COMPONENT_HEADING_RE.search(title)
    ):
        return False
    return True


def _semantic_heading_boundaries(
    text: str,
    lines: list[_Line],
    protected: list[tuple[int, int]],
) -> list[_Boundary]:
    """Split repeated authored headings while excluding answer-choice labels."""

    markdown: list[_Marker] = []
    bold: list[_Marker] = []
    for line in lines:
        if _is_protected(line.start, protected):
            continue
        value = line.text.rstrip("\r\n")
        markdown_match = _MARKDOWN_HEADING_RE.match(value)
        if markdown_match is not None and not markdown_match.group("indent"):
            if _is_semantic_heading(
                markdown_match.group("title"), require_process_label=False
            ):
                markdown.append(
                    _Marker(
                        position=line.start,
                        family="semantic_markdown_heading",
                        number=len(markdown) + 1,
                        heading_level=len(markdown_match.group("heading")),
                        reason="semantic_heading_sequence",
                        confidence="hard",
                    )
                )
            continue
        bold_match = _BOLD_HEADING_RE.match(value)
        if bold_match is not None and not bold_match.group("indent"):
            # Plain bold lines are much more ambiguous than Markdown headings:
            # a model often bolds an answer choice, quotation, or verdict even
            # though it is part of the surrounding paragraph.  Require an
            # explicit process label (or a compact structural component name)
            # before treating such a line as an authored section boundary.
            if _is_semantic_heading(
                bold_match.group("title"), require_process_label=True
            ):
                bold.append(
                    _Marker(
                        position=line.start,
                        family="semantic_bold_heading",
                        number=len(bold) + 1,
                        heading_level=0,
                        reason="semantic_heading_sequence",
                        confidence="hard",
                    )
                )

    valid_markdown: list[list[_Marker]] = []
    for heading_level in sorted({marker.heading_level for marker in markdown}):
        siblings = [
            marker for marker in markdown if marker.heading_level == heading_level
        ]
        complete = _complete_run_prefix(text, siblings, minimum=50)
        if len(complete) >= 2:
            valid_markdown.append(complete)
    if valid_markdown:
        shallowest = min(run[0].heading_level for run in valid_markdown)
        runs = [
            run for run in valid_markdown if run[0].heading_level == shallowest
        ]
    else:
        complete_bold = _complete_run_prefix(text, bold, minimum=60)
        runs = [complete_bold] if len(complete_bold) >= 2 else []

    boundaries: list[_Boundary] = []
    for run in runs:
        for index, marker in enumerate(run):
            if index == 0 and not _first_marker_boundary_allowed(
                text, marker.position, minimum=80
            ):
                continue
            boundaries.append(
                _Boundary(marker.position, marker.reason, marker.confidence)
            )
    return boundaries


def _balanced_boxes(text: str) -> list[tuple[int, int]]:
    boxes: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(_BOX_PREFIX, cursor)
        if start < 0:
            return boxes
        depth = 1
        position = start + len(_BOX_PREFIX)
        while position < len(text) and depth:
            if text[position] == "{" and (position == 0 or text[position - 1] != "\\"):
                depth += 1
            elif text[position] == "}" and (position == 0 or text[position - 1] != "\\"):
                depth -= 1
            position += 1
        if depth:
            return boxes
        boxes.append((start, position))
        cursor = position


def _final_answer_boundary(
    text: str,
    lines: list[_Line],
    protected: list[tuple[int, int]],
    final_box: tuple[int, int] | None,
) -> _Boundary | None:
    if final_box is None:
        return None
    candidates = [
        line.start
        for line in lines
        if line.start <= final_box[0]
        and not _is_protected(line.start, protected)
        and _FINAL_HEADING_RE.match(line.text)
    ]
    if not candidates:
        return None
    position = max(candidates)
    if position <= 0 or _substantive_length(text[position:final_box[1]]) < 4:
        return None
    return _Boundary(position, "final_answer_heading", "hard")


def _final_answer_paragraph_boundary(
    text: str,
    protected: list[tuple[int, int]],
    final_box: tuple[int, int] | None,
) -> _Boundary | None:
    """Split a short, self-contained concluding paragraph around the last box."""

    if final_box is None:
        return None
    separators = list(
        re.finditer(r"(?:\r?\n)[ \t]*(?:\r?\n)+", text[: final_box[0]])
    )
    if not separators:
        return None
    position = separators[-1].end()
    if position <= 0 or position >= final_box[0]:
        return None
    if _is_protected(position, protected):
        return None

    prefix = text[:position]
    if _substantive_length(prefix) < 80 or not _ends_semantic_unit(prefix):
        return None

    lead = text[position : final_box[0]]
    tail = text[final_box[1] :]
    if _substantive_length(tail) > 0:
        return None
    lead_length = _substantive_length(lead)
    box_only = lead_length <= 48
    cued = bool(_FINAL_PARAGRAPH_CUE_RE.match(lead.lstrip()))
    if not box_only and not (cued and lead_length <= 320):
        return None
    return _Boundary(position, "final_answer_paragraph", "hard")


def _deduplicate_boundaries(boundaries: list[_Boundary]) -> list[_Boundary]:
    by_position: dict[int, _Boundary] = {}
    for boundary in boundaries:
        previous = by_position.get(boundary.position)
        if previous is None or (
            previous.confidence == "soft" and boundary.confidence == "hard"
        ):
            by_position[boundary.position] = boundary
    return [by_position[position] for position in sorted(by_position)]


def split_text_steps(response: str) -> list[TextStep]:
    """Split a decoded response using only high-confidence semantic boundaries.

    The returned spans form an exact partition of ``response``.  If the text
    does not provide enough evidence for independent steps, the whole response
    is intentionally returned as one step.
    """

    response = str(response)
    if not response:
        return []
    lines = _lines(response)
    protected = _protected_ranges(response, lines)
    boxes = _balanced_boxes(response)
    final_box = boxes[-1] if boxes else None

    explicit = _explicit_boundaries(response, lines, protected)
    if explicit:
        candidates = explicit
    else:
        # Try authored structures from strongest to weakest.  Each detector
        # requires a repeated sibling sequence and complete, substantive
        # sections; an ordinary list or isolated emphasized phrase is not a
        # boundary by itself.
        candidates = []
        for detector in (
            _markdown_numbered_heading_boundaries,
            _bold_numbered_heading_boundaries,
            _numbered_reasoning_boundaries,
            _semantic_heading_boundaries,
        ):
            candidates = detector(response, lines, protected)
            if candidates:
                break

    final_boundary = _final_answer_boundary(response, lines, protected, final_box)
    if final_boundary is None:
        final_boundary = _final_answer_paragraph_boundary(
            response, protected, final_box
        )
    if final_boundary is not None:
        candidates.append(final_boundary)
    candidates = [
        boundary
        for boundary in _deduplicate_boundaries(candidates)
        if 0 < boundary.position < len(response)
        and not _is_protected(boundary.position, protected)
    ]

    starts = [0, *[boundary.position for boundary in candidates]]
    reasons = [_Boundary(0, "start", "hard"), *candidates]
    steps: list[TextStep] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(response)
        if start >= end:
            continue
        kind = (
            "final_answer"
            if final_box is not None and start <= final_box[0] < end
            else "reasoning"
        )
        reason = reasons[index]
        steps.append(
            TextStep(
                index=len(steps),
                char_start=start,
                char_end=end,
                text=response[start:end],
                kind=kind,
                boundary_reason=reason.reason,
                confidence=reason.confidence,
            )
        )

    if "".join(step.text for step in steps) != response:
        raise RuntimeError("Text steps do not exactly recover the source response.")
    return steps


def _decode(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    skip_special_tokens: bool,
) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )


def _find_original_token_boundary(
    tokenizer: Any,
    token_ids: list[int],
    decoded_text: str,
    char_target: int,
    *,
    minimum_token_end: int,
    minimum_char_end: int,
    skip_special_tokens: bool,
    cache: dict[int, str],
) -> tuple[int, int]:
    """Find the first original-ID prefix that safely covers a text boundary."""

    required_chars = max(int(char_target), int(minimum_char_end))

    def decode_prefix(end: int) -> str:
        if end not in cache:
            cache[end] = _decode(
                tokenizer,
                token_ids[:end],
                skip_special_tokens=skip_special_tokens,
            )
        return cache[end]

    low = max(0, int(minimum_token_end))
    high = len(token_ids)
    while low < high:
        middle = (low + high) // 2
        if len(decode_prefix(middle)) < required_chars:
            low = middle + 1
        else:
            high = middle
    approximate = low

    def candidates(start: int, end: int) -> list[tuple[int, int]]:
        found: list[tuple[int, int]] = []
        start = max(int(minimum_token_end), start)
        end = min(len(token_ids), end)
        for token_end in range(start, end + 1):
            prefix = decode_prefix(token_end)
            if len(prefix) >= required_chars and decoded_text.startswith(prefix):
                found.append((token_end, len(prefix)))
        return found

    found = candidates(approximate - 16, approximate + 16)
    if not found:
        found = candidates(approximate - 128, approximate + 128)
    if not found:
        found = candidates(int(minimum_token_end), len(token_ids))
    if not found:
        raise ValueError(
            "No prefix of the original response IDs covers character boundary "
            f"{char_target}."
        )
    return min(found, key=lambda item: (item[1], item[0]))


def split_rollout_steps(
    response_ids: Sequence[int],
    tokenizer: Any,
    *,
    skip_special_tokens: bool = True,
) -> tuple[str, list[TokenStep]]:
    """Decode and split original rollout IDs without ever re-tokenizing text.

    A token crossing a proposed character boundary belongs wholly to the
    preceding step.  Invisible trailing special tokens, such as EOS when
    ``skip_special_tokens=True``, remain outside the visible step spans.
    """

    original_ids = [int(token_id) for token_id in response_ids]
    response = _decode(
        tokenizer,
        original_ids,
        skip_special_tokens=skip_special_tokens,
    )
    text_steps = split_text_steps(response)
    if not text_steps:
        return response, []

    cache: dict[int, str] = {0: ""}
    token_steps: list[TokenStep] = []
    previous_token_end = 0
    previous_char_end = 0
    for text_step in text_steps:
        if text_step.char_end <= previous_char_end:
            continue
        token_end, char_end = _find_original_token_boundary(
            tokenizer,
            original_ids,
            response,
            text_step.char_end,
            minimum_token_end=previous_token_end,
            minimum_char_end=previous_char_end + 1,
            skip_special_tokens=skip_special_tokens,
            cache=cache,
        )
        if token_end <= previous_token_end or char_end <= previous_char_end:
            continue
        token_steps.append(
            TokenStep(
                index=len(token_steps),
                char_start=previous_char_end,
                char_end=char_end,
                token_start=previous_token_end,
                token_end=token_end,
                text=response[previous_char_end:char_end],
                kind=text_step.kind,
                boundary_reason=text_step.boundary_reason,
                confidence=text_step.confidence,
            )
        )
        previous_token_end = token_end
        previous_char_end = char_end

    if previous_char_end != len(response):
        raise RuntimeError(
            "Token-aligned steps do not cover the complete visible response: "
            f"{previous_char_end}/{len(response)} characters."
        )
    if "".join(step.text for step in token_steps) != response:
        raise RuntimeError("Token-aligned steps do not recover the decoded response.")
    return response, token_steps


__all__ = [
    "TextStep",
    "TokenStep",
    "split_rollout_steps",
    "split_text_steps",
]
