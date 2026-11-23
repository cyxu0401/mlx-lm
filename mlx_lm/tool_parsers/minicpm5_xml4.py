# Copyright © 2026 MLX Contributors

"""Read complete MiniCPM5 XML4 calls from a text stream."""

import ast
import json
import re
from dataclasses import replace

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


def _function_end(text, start):
    while boundary := _FUNCTION_BOUNDARY.search(text, start):
        if boundary.group() == _END:
            return boundary.start()
        end = text.find(_CDATA_END, boundary.end())
        if end == -1:
            return text.find(_END, boundary.end())
        start = end + len(_CDATA_END)
    return -1


def _param_value(body, start):
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
        value, pos = _param_value(body, match.end())
        if name not in arguments and (name in properties or not properties):
            arguments[name] = _convert_value(value, properties.get(name))
    return arguments


class MiniCPM5ToolParser:
    """Parse XML4 using the full-response and incremental detector rules."""

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
        self._saw_call = False

    def parse(self, text):
        normal = []
        calls = []
        pos = 0
        while (start := text.find(_START, pos)) != -1:
            opening = _FUNCTION.match(text, start)
            if opening is None:
                break
            close = _function_end(text, opening.end())
            if close == -1:
                break

            normal.append(text[pos:start])
            end = close + len(_END)
            name = opening[1].strip()
            if name not in self._schemas:
                normal.append(text[start:end])
            else:
                properties, required = self._schemas[name]
                arguments = _parse_arguments(text[opening.end() : close], properties)
                if required.issubset(arguments):
                    try:
                        json.dumps(arguments)
                    except TypeError:
                        return text, []
                    calls.append({"name": name, "arguments": arguments})
            pos = end

        normal.append(text[pos:])
        return "".join(normal), calls

    def feed(self, text):
        """Consume one complete block per chunk and keep the tail for the next chunk."""
        self._pending += text
        start = self._pending.find(_START)
        if start == -1:
            if any(self._pending.endswith(_START[:i]) for i in range(1, len(_START))):
                return "", []
            content = "" if self._saw_call else self._pending
            self._pending = ""
            return content, []

        close = _function_end(self._pending, start)
        if close == -1:
            content = self._pending[:start]
            self._pending = self._pending[start:]
            return content, []

        end = close + len(_END)
        content, calls = self.parse(self._pending[:end])
        self._pending = self._pending[end:]
        if calls:
            self._saw_call = True
        return content, calls


class XML4Reader:
    def __init__(self, initial_state, tools, *, streaming):
        from ..generate import TextStateMachine

        # The MiniCPM5 chat template determines the initial reasoning state.
        self._reader = TextStateMachine({"reasoning": [("</think>", "normal")]}).reader(
            state=initial_state
        )
        self._parser = MiniCPM5ToolParser(tools)
        self._streaming = streaming
        self._content = []

    def _parse(self, delta, *, final=False):
        if self._streaming:
            content, calls = self._parser.feed(delta.content)
        else:
            self._content.append(delta.content)
            if not final:
                return replace(delta, content="")
            content, calls = self._parser.parse("".join(self._content))
        return replace(delta, content=content, tool_calls=tuple(calls))

    def feed(self, text):
        return self._parse(self._reader.feed(text))

    def finish(self, text):
        return self._parse(self._reader.finish(text), final=True)
