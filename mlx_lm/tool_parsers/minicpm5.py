# Copyright © 2026 Apple Inc.
# Copyright © 2026 MLX Contributors

"""Tool call parser for OpenBMB MiniCPM5."""

import ast
import json

import regex as re

from .minimax_m2 import _convert_param_value_with_types, _extract_types_from_schema

_START = "<function"
_START_PREFIXES = tuple(_START[:i] for i in range(len(_START) - 1, 0, -1))
tool_call_start = "<function name="
tool_call_end = "</function>"
_CDATA = "<![CDATA["
_CDATA_END = "]]>"
_PARAM_END = "</param>"
_FUNCTION = re.compile(r"""<function\s+name\s*=\s*(['"])([^'"<>]+)\1[^<>]*>""")
_PARAM = re.compile(r"""<param\s+name=['"]([^'"]+)['"][^>]*>""")
_CDATA_PARAM = re.compile(_PARAM.pattern + r"\s*" + re.escape(_CDATA))
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


def _function_end(text, start, *, final):
    while (close := text.find(tool_call_end, start)) != -1:
        cdata = text.find(_CDATA, start, close)
        if cdata == -1:
            return close
        cdata_end = text.find(_CDATA_END, cdata + len(_CDATA))
        if cdata_end == -1:
            # A later chunk may still close CDATA around this tag.
            return close if final else -1

        # If a later call opened CDATA, recover before the first following call.
        recovery_close = None
        for opening in _FUNCTION.finditer(text, close + len(tool_call_end), cdata_end):
            if recovery_close is None:
                recovery_close = text.rfind(tool_call_end, close, opening.start())
            next_close = text.find(tool_call_end, opening.end(), cdata_end)
            body_end = cdata_end if next_close == -1 else next_close
            if _CDATA_PARAM.search(text, opening.end(), body_end):
                return recovery_close

        start = cdata_end + len(_CDATA_END)
    return -1


def _param_value(body, start):
    cdata_start = _WHITESPACE.match(body, start).end()
    if body.startswith(_CDATA, cdata_start):
        start = cdata_start + len(_CDATA)
        end = body.find(_CDATA_END, start)
        if end != -1:
            stop = _WHITESPACE.match(body, end + len(_CDATA_END)).end()
            if body.startswith(_PARAM_END, stop):
                stop += len(_PARAM_END)
            return body[start:end], stop

    boundary = _PARAM_BOUNDARY.search(body, start)
    if boundary is None:
        return body[start:].strip(), len(body)
    stop = boundary.end() if boundary.group() == _PARAM_END else boundary.start()
    return body[start : boundary.start()].strip(), stop


def _convert_value(value, schema):
    if not isinstance(schema, dict) or schema.get("type") == "string":
        return value
    if not isinstance(schema.get("type"), str):
        types = _extract_types_from_schema(schema)
        if types == ["string"]:
            return value
        return _convert_param_value_with_types(value, types)
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
    """Parse complete MiniCPM5 tool calls and preserve text between them."""

    def __init__(self, tools):
        self._schemas = {}
        for tool in tools or ():
            function = tool.get("function")
            if function and function.get("name"):
                schema = function.get("parameters")
                schema = schema if isinstance(schema, dict) else {}
                self._schemas[function["name"]] = (
                    _schema_properties(schema),
                    set(schema.get("required") or ()),
                )
        self._pending = ""

    def _parse_block(self, text, opening, close):
        name = opening[2].strip()
        literal = text[opening.start() : close + len(tool_call_end)]
        if name not in self._schemas:
            return literal, None
        properties, required = self._schemas[name]
        try:
            arguments = _parse_arguments(text[opening.end() : close], properties)
            if not required.issubset(arguments):
                return literal, None
            json.dumps(arguments, allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            return literal, None
        return "", {"name": name, "arguments": arguments}

    def _scan(self, text, *, final, reader=None):
        """Yield text, parsed call, and whether the text belongs to an XML block."""
        if not self._schemas:
            yield text, None, False
            return ""

        pos = 0
        while pos < len(text):
            start = text.find(_START, pos)
            if start == -1:
                tail = text[pos:]
                if not final:
                    for prefix in _START_PREFIXES:
                        if tail.endswith(prefix):
                            yield tail[: -len(prefix)], None, False
                            return prefix
                yield tail, None, False
                return ""

            # The caller consumes the prefix before we consult its state.
            yield text[pos:start], None, False
            opening = (
                _FUNCTION.match(text, start, partial=not final)
                if reader is None or reader.state == "normal"
                else None
            )
            if opening is None:
                yield _START, None, False
                pos = start + len(_START)
                continue
            if opening.partial:
                return text[start:]

            close = _function_end(text, opening.end(), final=final)
            if close == -1:
                if not final:
                    return text[start:]
                yield text[start:], None, True
                return ""

            content, call = self._parse_block(text, opening, close)
            yield content, call, True
            pos = close + len(tool_call_end)
        return ""

    def _advance(self, text, *, final, reader=None):
        self._pending = yield from self._scan(
            self._pending + text, final=final, reader=reader
        )

    @staticmethod
    def _collect(pieces):
        content, calls = [], []
        for text, call, _ in pieces:
            content.append(text)
            if call is not None:
                calls.append(call)
        return "".join(content), calls

    def parse(self, text):
        return self._collect(self._scan(text, final=True))

    def feed(self, text):
        return self._collect(self._advance(text, final=False))

    def finish(self, text=""):
        return self._collect(self._advance(text, final=True))


def parse_tool_call(text, tools=None):
    """Parse full calls or a single call with its boundary markers removed."""
    body = text.lstrip()
    if body.startswith(('"', "'")):
        text = tool_call_start + body + tool_call_end
    _, calls = MiniCPM5ToolParser(tools).parse(text)
    if not calls:
        raise ValueError("No valid MiniCPM5 tool call.")
    return calls[0] if len(calls) == 1 else calls


class MiniCPM5Reader:
    def __init__(self, reader, tools):
        self._reader = reader
        self._parser = MiniCPM5ToolParser(tools)

    def _read(self, text, *, final):
        from ..generate import TextStateMachine

        delta = TextStateMachine.Delta()
        for content, call, is_xml in self._parser._advance(
            text, final=final, reader=self._reader
        ):
            if is_xml:
                # A tool body may contain literal tokenizer control markers.
                delta += self._reader.finish()
                delta += TextStateMachine.Delta(
                    content=content, tool_calls=(call,) if call is not None else ()
                )
            else:
                delta += self._reader.feed(content)
        if final:
            delta += self._reader.finish()
        return delta

    def feed(self, text):
        return self._read(text, final=False)

    def finish(self, text=""):
        return self._read(text, final=True)
