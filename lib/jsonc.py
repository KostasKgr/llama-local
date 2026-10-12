"""Strict JSONC parsing with source spans for small, formatting-preserving edits."""
from dataclasses import dataclass, field
import json
import math
import re


@dataclass
class Node:
    value: object
    start: int
    end: int
    children: dict = field(default_factory=dict)
    last_end: int = 0


class Parser:
    def __init__(self, source):
        self.source = source
        self.pos = 0

    def skip(self):
        while self.pos < len(self.source):
            if self.source[self.pos].isspace():
                self.pos += 1
            elif self.source.startswith('//', self.pos):
                end = self.source.find('\n', self.pos)
                self.pos = len(self.source) if end < 0 else end
            elif self.source.startswith('/*', self.pos):
                end = self.source.find('*/', self.pos + 2)
                if end < 0:
                    raise ValueError('Unterminated JSONC comment')
                self.pos = end + 2
            else:
                break

    def take(self, char):
        self.skip()
        if not self.source.startswith(char, self.pos):
            raise ValueError(f'Expected {char!r} at offset {self.pos}')
        self.pos += 1

    def node(self):
        self.skip()
        start = self.pos
        if self.source.startswith(('{', '['), start):
            obj = self.source[start] == '{'
            close = '}' if obj else ']'
            self.pos += 1
            value = {} if obj else []
            children = {}
            last_end = self.pos
            self.skip()
            while not self.source.startswith(close, self.pos):
                if obj:
                    key = self.scalar()
                    if not isinstance(key.value, str):
                        raise ValueError('JSONC object keys must be strings')
                    if key.value in children:
                        raise ValueError(f'Duplicate JSONC key: {key.value}')
                    self.take(':')
                child = self.node()
                if obj:
                    value[key.value] = child.value
                    children[key.value] = child
                else:
                    value.append(child.value)
                last_end = child.end
                self.skip()
                if self.source.startswith(close, self.pos):
                    break
                self.take(',')
                self.skip()
            self.take(close)
            return Node(value, start, self.pos, children, last_end)
        return self.scalar()

    def scalar(self):
        self.skip()
        start = self.pos
        try:
            value, end = json.JSONDecoder().raw_decode(self.source, start)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSONC at offset {start}') from exc
        if isinstance(value, (dict, list)) or isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Invalid JSONC scalar')
        self.pos = end
        return Node(value, start, end)


def parse(source):
    parser = Parser(source)
    root = parser.node()
    parser.skip()
    if parser.pos != len(source) or not isinstance(root.value, dict):
        raise ValueError('Config must contain exactly one JSONC object')
    return root


def set_value(source, path, value):
    """Change one property, creating missing ancestors without touching siblings."""
    root = parse(source)
    node = root
    for index, key in enumerate(path):
        if not isinstance(node.value, dict):
            raise ValueError(f'Expected object at {path[:index]}')
        if key not in node.children:
            for remaining in reversed(path[index + 1:]):
                value = {remaining: value}
            line_start = source.rfind('\n', 0, node.start) + 1
            parent_indent = re.match(r'[ \t]*', source[line_start:]).group()
            indent = parent_indent + '  '
            if node.children:
                first = next(iter(node.children))
                # Derive key indentation from the first property, not its value.
                key_start = source.find(json.dumps(first), node.start + 1)
                prefix = source[source.rfind('\n', 0, key_start) + 1:key_start]
                if prefix and not prefix.strip():
                    indent = prefix
            newline = '\r\n' if '\r\n' in source else '\n'
            rendered = json.dumps(value, indent=2, ensure_ascii=False).replace('\n', newline + indent)
            tail = source[node.last_end:node.end - 1]
            # A comma may be separated from the last value by comments.
            check = Parser(tail)
            check.skip()
            has_comma = tail.startswith(',', check.pos)
            comma = ',' if node.children and not has_comma else ''
            insertion = newline + indent + json.dumps(key) + ': ' + rendered + newline + parent_indent
            result = source[:node.last_end] + comma + tail.rstrip(' \t\r\n') + insertion + source[node.end - 1:]
            parse(result)
            return result
        node = node.children[key]
    rendered = json.dumps(value, ensure_ascii=False)
    result = source[:node.start] + rendered + source[node.end:]
    parse(result)
    return result
