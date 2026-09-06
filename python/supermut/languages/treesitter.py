"""tree-sitter-backed Language implementations: JS, TS, Kotlin, Swift.

One generic implementation, per-language configuration. All four grammars
expose functions as `function_declaration` (plus `method_definition` in
JS/TS classes) with a `name` field — verified against the installed
grammars, not assumed.

Validation here is *parse-level* (exactly one named function, no errors);
interpreted-language mutants get their real check from the test run, and
compiled languages fail mutants at schemata build time.
"""

from __future__ import annotations

from dataclasses import dataclass

import tree_sitter

from supermut.languages.base import register
from supermut.mutate import FunctionTarget

__all__ = ["TreeSitterLanguage", "JAVASCRIPT", "TYPESCRIPT", "KOTLIN", "SWIFT"]

_FUNC_NODE_TYPES = ("function_declaration", "method_definition")


@dataclass
class TreeSitterLanguage:
    name: str
    extensions: tuple[str, ...]
    comment_prefix: str
    _language: tree_sitter.Language

    def _parser(self) -> tree_sitter.Parser:
        return tree_sitter.Parser(self._language)

    def _parse(self, source: str) -> tree_sitter.Tree:
        return self._parser().parse(source.encode())

    def find_targets(self, module_source: str) -> list[FunctionTarget]:
        src_bytes = module_source.encode()
        lines = module_source.split("\n")
        tree = self._parse(module_source)
        targets: list[FunctionTarget] = []

        def visit(node: tree_sitter.Node, inside_function: bool) -> None:
            for child in node.children:
                is_func = child.type in _FUNC_NODE_TYPES
                if is_func and not inside_function:
                    name_node = child.child_by_field_name("name")
                    if name_node is not None:
                        # An `export`(-default) wrapper shares the function's
                        # lines; the span must cover it and apply_mutant must
                        # put it back, or every mutant of an exported function
                        # silently breaks the module's imports.
                        prefix = ""
                        span = child
                        if node.type == "export_statement":
                            prefix = src_bytes[
                                node.start_byte : child.start_byte
                            ].decode()
                            span = node
                        if "\n" in prefix:
                            continue  # multi-line wrapper: skip, stay exact
                        start_line = span.start_point[0] + 1
                        end_line = max(span.end_point[0], child.end_point[0]) + 1
                        seg = src_bytes[child.start_byte : child.end_byte].decode()
                        indent = " " * span.start_point[1]
                        # keep the original indentation on the first line too,
                        # matching the Python frontend's padded segments
                        targets.append(
                            FunctionTarget(
                                name=name_node.text.decode(),
                                source=indent + seg,
                                start_line=start_line,
                                end_line=end_line,
                                indent=indent,
                                prefix=prefix,
                            )
                        )
                visit(child, inside_function or is_func)

        visit(tree.root_node, False)
        del lines
        return targets

    def validate(self, candidate: str, name: str) -> bool:
        tree = self._parse(candidate)
        if tree.root_node.has_error:
            return False
        funcs = [
            c
            for c in tree.root_node.children
            if c.type in _FUNC_NODE_TYPES
        ]
        if len(funcs) != 1:
            return False
        name_node = funcs[0].child_by_field_name("name")
        return name_node is not None and name_node.text.decode() == name

    def normalize(self, source: str) -> str | None:
        tree = self._parse(source)
        if tree.root_node.has_error:
            return None
        # S-expression drops comments and whitespace but keeps structure;
        # token text is appended for leaves so `a+1` != `a+2`.
        out: list[str] = []

        def sexp(node: tree_sitter.Node) -> None:
            # comment node types vary per grammar: comment, line_comment,
            # block_comment, multiline_comment
            if "comment" in node.type:
                return
            if node.child_count == 0:
                out.append(node.text.decode())
            else:
                out.append(f"({node.type}")
                for c in node.children:
                    sexp(c)
                out.append(")")

        sexp(tree.root_node)
        return " ".join(out)


def _load(module: str) -> tree_sitter.Language:
    m = __import__(module)
    if module == "tree_sitter_typescript":
        return tree_sitter.Language(m.language_typescript())
    return tree_sitter.Language(m.language())


JAVASCRIPT = register(
    TreeSitterLanguage(
        name="javascript",
        extensions=(".js", ".mjs", ".cjs", ".jsx"),
        comment_prefix="//",
        _language=_load("tree_sitter_javascript"),
    )
)
TYPESCRIPT = register(
    TreeSitterLanguage(
        name="typescript",
        extensions=(".ts", ".mts", ".tsx"),
        comment_prefix="//",
        _language=_load("tree_sitter_typescript"),
    )
)
KOTLIN = register(
    TreeSitterLanguage(
        name="kotlin",
        extensions=(".kt", ".kts"),
        comment_prefix="//",
        _language=_load("tree_sitter_kotlin"),
    )
)
SWIFT = register(
    TreeSitterLanguage(
        name="swift",
        extensions=(".swift",),
        comment_prefix="//",
        _language=_load("tree_sitter_swift"),
    )
)
