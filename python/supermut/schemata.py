"""Mutation schemata assembly for compiled languages (Kotlin, Swift).

File-swap testing is priced out for compiled languages (one build per
mutant), so all mutants of a file compile in ONCE behind a runtime
switch (Untch/Offutt/Harrold '93; Muter and Stryker.NET are the living
precedents). An injected helper reads ``SUPERMUT_MUTANT`` exactly once
at process start; the harness flips mutants by re-running tests with
the env var set — no rebuilds. ``canary_id`` activates a failure branch
in every dispatcher at once (the canary gate's schemata analogue).

Shapes (from reading Muter/Stryker/PIT source, 2026-09-06):
- Kotlin: the original function becomes a dispatcher keeping its exact
  header (modifiers, receiver, defaults); the original body and each
  mutant become file/class-private sibling functions called
  positionally. Kotlin needs no argument labels, which keeps
  forwarding trivial.
- Swift: variants are functions NESTED inside the original function,
  followed by a switch; nesting preserves protocol witness tables,
  @objc selectors, and the overload set, and the ``default: break``
  branch falls through into the untouched original body. Calls forward
  argument labels/inout explicitly.

Functions the schemata cannot host are skipped with a reason (Kotlin
``inline`` is already inlined at call sites; Swift ``mutating`` bodies
can't move into nested funcs; result builders reshape bodies; variadics
can't be re-forwarded in Swift). Skipped mutants surface as
UNSUPPORTED, never as kills.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tree_sitter

from supermut.mutate import Mutant, _dedent, _reindent

__all__ = ["Schemata", "build_schemata", "failed_ids", "SUPERMUT_ENV"]

SUPERMUT_ENV = "SUPERMUT_MUTANT"

# Variant blocks are fenced with these comments so compiler diagnostics
# (file:line) map back to a mutant id when the schemata build fails.
_BEGIN = "// __supermut_begin_{id}"
_END = "// __supermut_end_{id}"

_KOTLIN_HELPER = """\
{package}internal object __Supermut {{
    val active: Int = System.getenv("SUPERMUT_MUTANT")?.toIntOrNull() ?: 0
}}
"""

_SWIFT_HELPER = """\
import Foundation

enum __Supermut {
    static let active: Int =
        Int(ProcessInfo.processInfo.environment["SUPERMUT_MUTANT"] ?? "") ?? 0
}
"""

# Modifiers a variant copy must not carry: it is a new private symbol,
# not the original API surface.
_KT_STRIP_MODIFIERS = (
    "public", "internal", "protected", "private",
    "override", "operator", "infix", "tailrec",
)
_KT_SKIP_MODIFIERS = ("inline", "external", "expect", "abstract")
_SWIFT_STRIP_MODIFIERS = ("public", "internal", "fileprivate", "private",
                          "open", "override", "static", "class", "final")
_SWIFT_SKIP_ATTRS = ("inlinable", "inline", "_cdecl", "_silgen_name",
                     "ViewBuilder", "resultBuilder", "SceneBuilder")


@dataclass
class Schemata:
    """One file's worth of compiled-in mutants behind the runtime switch."""

    source: str  # whole module with dispatchers spliced in
    helper_filename: str
    helper_source: str
    by_id: dict[int, Mutant] = field(default_factory=dict)
    skipped: list[tuple[Mutant, str]] = field(default_factory=list)
    canary_id: int = 0


def _func_node(language, source: str) -> tree_sitter.Node | None:
    """The single top-level function node of a standalone function source."""
    tree = tree_sitter.Parser(language._language).parse(source.encode())
    if tree.root_node.has_error:
        return None
    for child in tree.root_node.children:
        if child.type in ("function_declaration", "method_definition"):
            return child
    return None


def _node_text(source_bytes: bytes, node: tree_sitter.Node) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode()


def _rename(source: str, func: tree_sitter.Node, new_name: str) -> str:
    name = func.child_by_field_name("name")
    b = source.encode()
    return (b[: name.start_byte] + new_name.encode() + b[name.end_byte :]).decode()


def _strip_modifiers(source: str, func: tree_sitter.Node, names: tuple[str, ...]) -> str:
    """Remove listed modifier keywords from the function's modifier list."""
    b = source.encode()
    spans: list[tuple[int, int]] = []
    for child in func.children:
        if child.type != "modifiers":
            continue
        for mod in child.children:
            if _node_text(b, mod).strip() in names:
                spans.append((mod.start_byte, mod.end_byte))
    for start, end in sorted(spans, reverse=True):
        # eat one following space/newline so no blank gap remains
        while end < len(b) and b[end : end + 1] in (b" ", b"\n"):
            end += 1
        b = b[:start] + b[end:]
    return b.decode()


# --- Kotlin -----------------------------------------------------------------


def _kt_params(func: tree_sitter.Node, b: bytes) -> list[tuple[str, bool]]:
    """[(name, is_vararg)] — vararg sits in a parameter_modifiers sibling
    *preceding* its parameter inside function_value_parameters."""
    params: list[tuple[str, bool]] = []
    for child in func.children:
        if child.type != "function_value_parameters":
            continue
        pending_vararg = False
        for c in child.children:
            if c.type == "parameter_modifiers":
                pending_vararg = "vararg" in _node_text(b, c)
            elif c.type == "parameter":
                name = next(
                    _node_text(b, g) for g in c.children if g.type == "identifier"
                )
                params.append((name, pending_vararg))
                pending_vararg = False
    return params


def _kt_skip_reason(func: tree_sitter.Node, b: bytes) -> str | None:
    for child in func.children:
        if child.type == "modifiers":
            text = _node_text(b, child)
            for kw in _KT_SKIP_MODIFIERS:
                if kw in text.split():
                    return f"kotlin `{kw}` functions can't host a runtime switch"
    return None


def _kt_dispatcher(
    dedented: str,
    func: tree_sitter.Node,
    ids: list[int],
    canary_id: int,
) -> str:
    b = dedented.encode()
    name = _node_text(b, func.child_by_field_name("name"))
    body = next(c for c in func.children if c.type == "function_body")
    header = b[: body.start_byte].decode().rstrip()
    args = ", ".join(("*" + n) if v else n for n, v in _kt_params(func, b))
    branches = [f"        {i} -> {name}__sm_{i}({args})" for i in ids]
    branches.append(
        f'        {canary_id} -> throw RuntimeException("supermut canary")'
    )
    branches.append(f"        else -> {name}__sm_orig({args})")
    when = "when (__Supermut.active) {\n" + "\n".join(branches) + "\n    }"
    expression_style = body.children and body.children[0].type == "="
    if expression_style:
        return f"{header} = {when}"
    return f"{header} {{\n    return {when}\n}}"


def _kt_variant(source: str, func: tree_sitter.Node, new_name: str) -> str:
    return "private " + _strip_modifiers(
        _rename(source, func, new_name), func, _KT_STRIP_MODIFIERS
    )


def _build_kotlin_block(
    language, dedented: str, variants: list[tuple[int, str]], canary_id: int
) -> tuple[str | None, list[tuple[int, str]], str | None]:
    """(replacement text, accepted (id, source) pairs, skip reason)."""
    func = _func_node(language, dedented)
    if func is None:
        return None, [], "original function does not parse standalone"
    reason = _kt_skip_reason(func, dedented.encode())
    if reason:
        return None, [], reason
    name = _node_text(dedented.encode(), func.child_by_field_name("name"))

    accepted: list[tuple[int, str]] = []
    parts: list[str] = []
    for mid, msource in variants:
        mfunc = _func_node(language, msource)
        if mfunc is None:
            continue
        parts.append(
            _BEGIN.format(id=mid)
            + "\n"
            + _kt_variant(msource, mfunc, f"{name}__sm_{mid}")
            + "\n"
            + _END.format(id=mid)
        )
        accepted.append((mid, msource))
    dispatcher = _kt_dispatcher(
        dedented, func, [mid for mid, _ in accepted], canary_id
    )
    orig = _kt_variant(dedented, func, f"{name}__sm_orig")
    return "\n\n".join([dispatcher, orig, *parts]), accepted, None


# --- Swift ------------------------------------------------------------------


def _swift_params(func: tree_sitter.Node, b: bytes) -> list[str] | None:
    """Forwarding argument list; None when a parameter can't be forwarded
    (variadic ``...``)."""
    args: list[str] = []
    for child in func.children:
        if child.type != "parameter":
            continue
        if any(g.type == "..." for g in child.children):
            return None
        external = child.child_by_field_name("external_name")
        internal = child.child_by_field_name("name")
        name = _node_text(b, internal)
        label = _node_text(b, external) if external is not None else name
        inout = "inout" in {
            _node_text(b, g).strip() for g in child.children
            if g.type == "parameter_modifiers"
        }
        value = ("&" if inout else "") + name
        args.append(value if label == "_" else f"{label}: {value}")
    return args


def _swift_skip_reason(func: tree_sitter.Node, b: bytes) -> str | None:
    for child in func.children:
        if child.type == "modifiers":
            text = _node_text(b, child)
            if "mutating" in text.split():
                return "swift `mutating` bodies can't move into nested functions"
            for attr in _SWIFT_SKIP_ATTRS:
                if f"@{attr}" in text:
                    return f"swift @{attr} functions can't host a runtime switch"
    if _swift_params(func, b) is None:
        return "swift variadic parameters can't be re-forwarded"
    return None


def _swift_variant(source: str, func: tree_sitter.Node, new_name: str) -> str:
    return _strip_modifiers(
        _rename(source, func, new_name), func, _SWIFT_STRIP_MODIFIERS
    )


def _build_swift_block(
    language, dedented: str, variants: list[tuple[int, str]], canary_id: int
) -> tuple[str | None, list[tuple[int, str]], str | None]:
    func = _func_node(language, dedented)
    if func is None:
        return None, [], "original function does not parse standalone"
    b = dedented.encode()
    reason = _swift_skip_reason(func, b)
    if reason:
        return None, [], reason
    args = _swift_params(func, b)
    effects = {c.type for c in func.children} & {"async", "throws", "rethrows"}
    call_prefix = ("try " if effects & {"throws", "rethrows"} else "") + (
        "await " if "async" in effects else ""
    )
    body = next(
        c for c in func.children if c.type == "function_body"
    )
    open_brace = next(c for c in body.children if c.type == "{")

    accepted: list[tuple[int, str]] = []
    nested: list[str] = []
    cases: list[str] = []
    for mid, msource in variants:
        mfunc = _func_node(language, msource)
        if mfunc is None:
            continue
        nested.append(
            _reindent(
                _BEGIN.format(id=mid)
                + "\n"
                + _swift_variant(msource, mfunc, f"__sm_{mid}")
                + "\n"
                + _END.format(id=mid),
                "    ",
            )
        )
        cases.append(
            f"    case {mid}: return {call_prefix}__sm_{mid}({', '.join(args)})"
        )
        accepted.append((mid, msource))
    cases.append(f'    case {canary_id}: fatalError("supermut canary")')
    switch = (
        "    switch __Supermut.active {\n"
        + "\n".join(cases)
        + "\n    default: break\n    }"
    )
    injected = "\n" + "\n".join(nested) + "\n" + switch
    out = (b[: open_brace.end_byte] + injected.encode() + b[open_brace.end_byte :]).decode()
    return out, accepted, None


# --- entry point ------------------------------------------------------------

_BUILDERS = {
    "kotlin": (_build_kotlin_block, "__supermut_helper.kt"),
    "swift": (_build_swift_block, "__supermut_helper.swift"),
}


def build_schemata(module_source: str, mutants: list[Mutant], language) -> Schemata:
    """Splice every mutant into one compilable module behind the switch.

    Mutant ids are 1-based positions in ``mutants``; ``canary_id`` is one
    past the last id. Unbuildable targets/mutants land in ``skipped``.
    """
    builder, helper_name = _BUILDERS[language.name]

    by_target: dict[tuple[int, str], list[tuple[int, Mutant]]] = {}
    for i, m in enumerate(mutants):
        key = (m.target.start_line, m.target.name)
        by_target.setdefault(key, []).append((i + 1, m))
    canary_id = len(mutants) + 1

    schemata = Schemata(
        source=module_source,
        helper_filename=helper_name,
        helper_source=_helper_source(module_source, language),
        canary_id=canary_id,
    )
    out = module_source
    # bottom-up so earlier line spans stay valid, mirroring _apply_many
    for key in sorted(by_target, reverse=True):
        entries = by_target[key]
        target = entries[0][1].target
        dedented = _dedent(target.source, target.indent)
        variants = [
            (mid, _dedent(m.source, target.indent)) for mid, m in entries
        ]
        block, accepted, reason = builder(language, dedented, variants, canary_id)
        if reason is not None:
            schemata.skipped += [(m, reason) for _, m in entries]
            continue
        accepted_ids = {mid for mid, _ in accepted}
        for mid, m in entries:
            if mid in accepted_ids:
                schemata.by_id[mid] = m
            else:
                schemata.skipped.append((m, "mutant does not parse standalone"))
        # Reuse apply_mutant so target.prefix (if any) is restored.
        from supermut.harness import apply_mutant

        out = apply_mutant(
            out,
            Mutant(target=target, source=_reindent(block, target.indent)),
        )
    schemata.source = out
    return schemata


class SchemataBuildError(RuntimeError):
    """The schemata build failed; the message carries compiler diagnostics."""


def failed_ids(schemata_source: str, diagnostics: str, file_name: str) -> set[int]:
    """Mutant ids whose fenced variant block owns a compile error.

    Both swiftc and kotlinc emit ``path/file:line:col: error: …``; any
    error line falling inside a ``__supermut_begin/end`` fence names its
    mutant. Errors outside every fence return an empty set — the caller
    must surface those instead of dropping mutants blindly.
    """
    enclosing: dict[int, int] = {}
    current: int | None = None
    for lineno, line in enumerate(schemata_source.split("\n"), 1):
        s = line.strip()
        if s.startswith("// __supermut_begin_"):
            current = int(s.rsplit("_", 1)[1])
        if current is not None:
            enclosing[lineno] = current
        if s.startswith("// __supermut_end_"):
            current = None
    pattern = re.compile(re.escape(file_name) + r":(\d+)(?::\d+)?: *error")
    return {
        enclosing[int(m.group(1))]
        for m in pattern.finditer(diagnostics)
        if int(m.group(1)) in enclosing
    }


def _helper_source(module_source: str, language) -> str:
    if language.name == "swift":
        return _SWIFT_HELPER
    package = ""
    for line in module_source.split("\n"):
        if line.startswith("package "):
            package = line.strip() + "\n\n"
            break
        if line.strip() and not line.startswith(("//", "@file", "import")):
            break
    return _KOTLIN_HELPER.format(package=package)
