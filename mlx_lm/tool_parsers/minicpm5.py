# Copyright © 2026 Apple Inc.

"""
Tool call parser for OpenBMB MiniCPM5.

The chat template asks the model for
``<function name="fn"><param name="p">value</param></function>`` with no
outer wrapper. Values containing ``<``, ``&`` or newlines are wrapped in a
CDATA block, and parallel calls are consecutive ``<function>`` blocks.
"""

import ast
import json
from typing import Any, Optional

import regex as re

from .minimax_m2 import (
    _convert_param_value_with_types,
    _extract_name,
    _get_param_types_from_config,
)

tool_call_start = "<function name="
tool_call_end = "</function>"

# The state machine strips both markers, so the server hands the parser
# ``"fn"><param ...>...``; the leading ``<function name=`` is optional here
# so the same parser also accepts the full text.
_function_regex = re.compile(
    r"(?:<function\s+)?(?:name\s*=\s*)?"
    r"(?P<name>\"[^\"]*\"|'[^']*'|[^\s\"'<>]+)\s*>"
    r"(?P<body>.*?)(?:</function>|$)",
    re.DOTALL,
)
_param_regex = re.compile(
    r"<param\s+name\s*=\s*(?P<name>\"[^\"]*\"|'[^']*'|[^\s\"'<>]+)\s*>"
    r"(?P<value>.*?)</param>",
    re.DOTALL,
)
_cdata_regex = re.compile(r"^\s*<!\[CDATA\[(.*?)\]\]>\s*$", re.DOTALL)


def _param_value(raw: str) -> str:
    if (match := _cdata_regex.match(raw)) is not None:
        return match.group(1)
    return raw.strip()


def parse_tool_call(model_output: str, tools: Optional[Any] = None):
    function_matches = list(_function_regex.finditer(model_output))
    if not function_matches:
        raise ValueError("No function provided.")

    param_config_for = {}
    for tool in tools or []:
        if function := tool.get("function", False):
            if params := function.get("parameters", False):
                param_config_for[function["name"]] = params.get("properties", {})

    calls = []
    for function_match in function_matches:
        function_name = _extract_name(function_match.group("name"))
        param_config = param_config_for.get(function_name, {})
        arguments = {}
        for param_match in _param_regex.finditer(function_match.group("body")):
            param_name = _extract_name(param_match.group("name"))
            arguments[param_name] = _convert_param_value_with_types(
                _param_value(param_match.group("value")),
                _get_param_types_from_config(param_name, param_config),
            )
        calls.append({"name": function_name, "arguments": arguments})

    if len(calls) == 1:
        return calls[0]
    return calls


_START = "<function"
_END = "</function>"
_CDATA = "<![CDATA["
_CDATA_END = "]]>"
_PARAM_END = "</param>"
_FUNCTION = re.compile(r"""<function\s+name=['"]([^'"]+)['"][^>]*>""")
_PARAM = re.compile(r"""<param\s+name=['"]([^'"]+)['"][^>]*>""")
_FUNCTION_BOUNDARY = re.compile(r"</function>|<!\[CDATA\[")
_PARAM_BOUNDARY = re.compile(r"</param>|<param")
_WHITESPACE = re.compile(r"\s*")


def _schema_properties(schema):
    properties = schema.get("properties")
    if isinstance(properties, dict):
        return properties
    properties = {}
    for keyword in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(keyword)
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if isinstance(branch, dict):
                for name, spec in _schema_properties(branch).items():
                    properties.setdefault(name, spec)
    return properties


def _function_end(text, start, final):
    while boundary := _FUNCTION_BOUNDARY.search(text, start):
        if boundary.group() == _END:
            return boundary.start()
        end = text.find(_CDATA_END, boundary.end())
        if end == -1:
            # Resolve broken CDATA at end of generation, after all chunks arrive.
            return text.find(_END, boundary.end()) if final else -1
        start = end + len(_CDATA_END)
    return -1


def _xml4_param_value(body, start):
    if body.startswith(_CDATA, start):
        start += len(_CDATA)
        end = body.find(_CDATA_END, start)
        if end != -1:
            stop = end + len(_CDATA_END)
            if body.startswith(_PARAM_END, stop):
                stop += len(_PARAM_END)
            return body[start:end], stop

    boundary = _PARAM_BOUNDARY.search(body, start)
    if boundary is None:
        return body[start:].strip(), len(body)
    stop = boundary.end() if boundary.group() == _PARAM_END else boundary.start()
    return body[start : boundary.start()].strip(), stop


def _convert_value(value, schema):
    if not isinstance(schema, dict) or schema.get("type") in (None, "string"):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value


def _parse_arguments(body, properties):
    arguments = {}
    pos = 0
    while pos < len(body):
        pos = _WHITESPACE.match(body, pos).end()
        match = _PARAM.match(body, pos)
        if match is None:
            if not body.startswith("<param", pos):
                break
            end = body.find(_PARAM_END, pos)
            if end == -1:
                break
            pos = end + len(_PARAM_END)
            continue
        name = match[1].strip()
        value, pos = _xml4_param_value(body, match.end())
        if name not in arguments and (name in properties or not properties):
            arguments[name] = _convert_value(value, properties.get(name))
    return arguments


class MiniCPM5ToolParser:
    """Parse XML4 after reasoning separation, emitting each complete call once."""

    def __init__(self, tools):
        self._schemas = {}
        for tool in tools:
            function = tool.get("function")
            if function and function.get("name"):
                schema = function.get("parameters")
                schema = schema if isinstance(schema, dict) else {}
                self._schemas[function["name"]] = (
                    _schema_properties(schema),
                    set(schema.get("required") or ()),
                )
        self._pending = ""
        self._text_only = False

    def feed(self, text, *, final=False):
        if self._text_only:
            return text, []
        self._pending += text
        normal = []
        calls = []
        while self._pending:
            start = self._pending.find(_START)
            if start == -1:
                # Retain a partial opening marker across stream chunks.
                end = len(self._pending)
                if not final:
                    for length in range(len(_START) - 1, 0, -1):
                        if self._pending.endswith(_START[:length]):
                            end -= length
                            break
                normal.append(self._pending[:end])
                self._pending = self._pending[end:]
                break

            normal.append(self._pending[:start])
            self._pending = self._pending[start:]
            opening = _FUNCTION.match(self._pending)
            if opening is None:
                if ">" in self._pending:
                    self._text_only = True
                    final = True
                break
            close = _function_end(self._pending, opening.end(), final)
            if close == -1:
                break

            end = close + len(_END)
            name = opening[1].strip()
            if name not in self._schemas:
                normal.append(self._pending[:end])
            else:
                properties, required = self._schemas[name]
                arguments = _parse_arguments(
                    self._pending[opening.end() : close], properties
                )
                if required.issubset(arguments):
                    try:
                        json.dumps(arguments)
                    except TypeError:
                        normal.append(self._pending[:end])
                    else:
                        calls.append({"name": name, "arguments": arguments})
            self._pending = self._pending[end:]

        if final:
            normal.append(self._pending)
            self._pending = ""
        return "".join(normal), calls
