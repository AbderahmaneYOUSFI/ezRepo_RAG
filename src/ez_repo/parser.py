"""Parse Python source with Tree-sitter to extract classes, functions, imports and calls.

Output is plain dataclasses, ready to be written to Neo4j:

- nodes: files (id = relative path), symbols (id = "<path>::<qualified name>"),
  external modules (id = "module:<dotted name>");
- relations: DEFINES, IMPORTS, CALLS, INHERITS.

CALLS and INHERITS are only emitted when the target is resolved without guessing:
a bare name bound exactly once to a definition (same file, enclosing function, or a
`from x import name` pointing into the repository), or `self.method()` where the method
is defined in the enclosing class. Everything else (attributes, builtins, dynamic
calls, shadowed names) is kept as an unresolved Reference and produces no relation.
"""

import inspect
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from ez_repo.ingest import SourceFile

PY_LANGUAGE = Language(tree_sitter_python.language())

# A name binding: a symbol id, a "from module import name" target, or None (unknown value).
Binding = str | tuple[str, str] | None


@dataclass(frozen=True)
class Symbol:
    id: str
    kind: str  # "class" | "function" | "method"
    name: str
    qualified_name: str  # within its module, e.g. "Greeter.greet"
    path: str
    start_line: int  # 1-based, decorators included
    end_line: int
    source: str
    signature: str
    docstring: str | None
    parent_id: str | None  # enclosing class or function; None at module level
    decorators: tuple[str, ...] = ()
    bases: tuple[str, ...] = ()  # base classes as written (classes only)


@dataclass(frozen=True)
class Import:
    path: str
    module: str  # as written, without leading dots ("" for "from . import x")
    name: str | None  # imported name for "from" imports, "*" for a wildcard
    alias: str | None
    level: int  # number of leading dots of a relative import
    resolved_module: str | None  # absolute dotted module; None if outside the repository root
    line: int


@dataclass(frozen=True)
class Reference:
    """A call or a base class as found in the code, before cross-file resolution."""

    relation: str  # "CALLS" | "INHERITS"
    source_id: str
    name: str  # as written: "helper", "self.save", "Base"
    line: int
    target_id: str | None = None  # resolved within the file
    via_import: tuple[str, str] | None = None  # (absolute module, name), resolved across files


@dataclass
class ParsedFile:
    path: str
    module: str  # dotted module name derived from the path
    has_errors: bool  # Tree-sitter found syntax errors (extraction is best effort)
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    bindings: dict[str, Binding] = field(default_factory=dict)  # module-level names


@dataclass(frozen=True)
class Relation:
    type: str  # "DEFINES" | "IMPORTS" | "CALLS" | "INHERITS"
    source_id: str
    target_id: str
    line: int | None = None


@dataclass
class CodeGraph:
    files: list[ParsedFile]
    relations: list[Relation]

    @property
    def symbols(self) -> list[Symbol]:
        return [s for f in self.files for s in f.symbols]

    @property
    def imports(self) -> list[Import]:
        return [i for f in self.files for i in f.imports]

    @property
    def external_modules(self) -> list[str]:
        prefix = "module:"
        return sorted(
            {r.target_id[len(prefix) :] for r in self.relations if r.target_id.startswith(prefix)}
        )


def module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def parse_file(path: str, text: str) -> ParsedFile:
    return _FileParser(path, text).parse()


def parse_repository(files: Iterable[SourceFile]) -> CodeGraph:
    parsed = [parse_file(f.path, f.text) for f in files]
    return CodeGraph(files=parsed, relations=build_relations(parsed))


def build_relations(files: list[ParsedFile]) -> list[Relation]:
    index = _module_index(files)
    kinds = {s.id: s.kind for f in files for s in f.symbols}
    relations: dict[tuple[str, str, str], Relation] = {}

    def add(rel: Relation) -> None:
        relations.setdefault((rel.type, rel.source_id, rel.target_id), rel)

    def resolve_import(module: str, name: str, depth: int = 0) -> str | None:
        target = index.get(module)
        if target is None or depth > 5:
            return None
        binding = target.bindings.get(name)
        if isinstance(binding, tuple):  # re-export, e.g. "from .core import name" in __init__
            return resolve_import(*binding, depth=depth + 1)
        return binding

    for f in files:
        for s in f.symbols:
            add(Relation("DEFINES", s.parent_id or f.path, s.id))

        for imp in f.imports:
            if imp.resolved_module is None:
                continue
            submodule = f"{imp.resolved_module}.{imp.name}" if imp.name else None
            target = index.get(submodule) or index.get(imp.resolved_module)
            if target is not None:
                add(Relation("IMPORTS", f.path, target.path, imp.line))
            elif imp.level == 0:
                add(Relation("IMPORTS", f.path, f"module:{imp.resolved_module}", imp.line))

        for ref in f.references:
            target_id = ref.target_id
            if target_id is None and ref.via_import:
                target_id = resolve_import(*ref.via_import)
            if target_id is None or target_id not in kinds:
                continue
            if ref.relation == "INHERITS" and kinds[target_id] != "class":
                continue
            add(Relation(ref.relation, ref.source_id, target_id, ref.line))

    return list(relations.values())


def _module_index(files: list[ParsedFile]) -> dict[str, ParsedFile]:
    """Map dotted module names to files; ambiguous names are dropped."""
    candidates: dict[str, list[ParsedFile]] = defaultdict(list)
    for f in files:
        names = {f.module}
        if f.module.startswith("src."):  # src/ layout: "src/pkg/mod.py" is imported as pkg.mod
            names.add(f.module[len("src.") :])
        for name in names:
            if name:
                candidates[name].append(f)
    return {name: fs[0] for name, fs in candidates.items() if len(fs) == 1}


class _Scope:
    def __init__(self, kind: str, symbol: Symbol | None = None, parent: "_Scope | None" = None):
        self.kind = kind  # "module" | "class" | "function"
        self.symbol = symbol
        self.parent = parent
        self.bindings: dict[str, list[Binding]] = defaultdict(list)
        self.self_name: str | None = None  # first parameter of a regular method

    def bind(self, name: str, value: Binding = None) -> None:
        self.bindings[name].append(value)

    def lookup(self, name: str) -> tuple[bool, Binding]:
        """Return (found, value). A name bound several times resolves to None."""
        if name not in self.bindings:
            return False, None
        values = self.bindings[name]
        return True, values[0] if len(values) == 1 else None


_DEFINITIONS = {"function_definition", "class_definition"}
_ASSIGNMENTS = {"assignment", "augmented_assignment", "for_statement", "for_in_clause"}


class _FileParser:
    def __init__(self, path: str, text: str):
        self.path = path
        self.src = text.encode("utf-8")
        self.module = module_name(path)
        self.symbols: list[Symbol] = []
        self.imports: list[Import] = []
        self.ids: set[str] = set()
        # (relation, source symbol, name as written, line, scope to resolve from)
        self.raw_refs: list[tuple[str, Symbol, str, int, _Scope]] = []

    def parse(self) -> ParsedFile:
        tree = Parser(PY_LANGUAGE).parse(self.src)
        module_scope = _Scope("module")
        self._visit(tree.root_node, module_scope)
        return ParsedFile(
            path=self.path,
            module=self.module,
            has_errors=tree.root_node.has_error,
            symbols=self.symbols,
            imports=self.imports,
            references=[self._resolve(*raw) for raw in self.raw_refs],
            bindings={name: module_scope.lookup(name)[1] for name in module_scope.bindings},
        )

    # --- traversal -------------------------------------------------------------------

    def _visit(self, node: Node, scope: _Scope) -> None:
        for child in node.named_children:
            kind = child.type
            if kind == "decorated_definition":
                self._definition(child.child_by_field_name("definition"), child, scope)
            elif kind in _DEFINITIONS:
                self._definition(child, child, scope)
            elif kind in {"import_statement", "import_from_statement"}:
                self._import(child, scope)
            else:
                if kind in _ASSIGNMENTS:
                    self._bind_targets(child.child_by_field_name("left"), scope)
                elif kind == "named_expression":
                    self._bind_targets(child.child_by_field_name("name"), scope)
                elif kind == "as_pattern":
                    self._bind_targets(child.child_by_field_name("alias"), scope)
                elif kind == "call" and scope.kind == "function":
                    self._call(child, scope)
                self._visit(child, scope)

    def _definition(self, node: Node, outer: Node, scope: _Scope) -> None:
        name = self._text(node.child_by_field_name("name"))
        is_class = node.type == "class_definition"
        parent = scope.symbol
        qualified = f"{parent.qualified_name}.{name}" if parent else name
        decorators = tuple(
            self._text(c).lstrip("@").strip() for c in outer.children if c.type == "decorator"
        )

        if is_class:
            kind = "class"
            superclasses = node.child_by_field_name("superclasses")
            base_nodes = [
                c for c in (superclasses.named_children if superclasses else [])
                if c.type in {"identifier", "attribute"}
            ]  # fmt: skip
            bases = tuple(self._text(c) for c in base_nodes)
            signature = f"class {name}{self._text(superclasses) if superclasses else ''}"
        else:
            kind = "method" if scope.kind == "class" else "function"
            base_nodes, bases = [], ()
            params = node.child_by_field_name("parameters")
            returns = node.child_by_field_name("return_type")
            is_async = any(c.type == "async" for c in node.children)
            signature = f"{'async ' if is_async else ''}def {name}{self._text(params)}"
            if returns:
                signature += f" -> {self._text(returns)}"

        symbol = Symbol(
            id=self._unique_id(f"{self.path}::{qualified}", outer),
            kind=kind,
            name=name,
            qualified_name=qualified,
            path=self.path,
            start_line=outer.start_point[0] + 1,
            end_line=outer.end_point[0] + 1,
            source=self._text(outer),
            signature=signature,
            docstring=self._docstring(node),
            parent_id=parent.id if parent else None,
            decorators=decorators,
            bases=bases,
        )
        self.symbols.append(symbol)
        scope.bind(name, symbol.id)

        for base in base_nodes:
            if base.type == "identifier":
                self.raw_refs.append(("INHERITS", symbol, self._text(base), _line(base), scope))

        inner = _Scope("class" if is_class else "function", symbol, scope)
        if not is_class:
            param_names = self._parameter_names(node.child_by_field_name("parameters"))
            for param in param_names:
                inner.bind(param)
            if kind == "method" and param_names and "staticmethod" not in decorators:
                inner.self_name = param_names[0]
        self._visit(node.child_by_field_name("body"), inner)

    def _import(self, node: Node, scope: _Scope) -> None:
        line = _line(node)
        if node.type == "import_statement":
            for item in node.children_by_field_name("name"):
                module, alias = self._name_and_alias(item)
                self._add_import(module, None, alias, 0, module, line)
                scope.bind(alias or module.split(".")[0])
            return

        module_node = node.child_by_field_name("module_name")
        level, module = 0, self._text(module_node)
        if module_node.type == "relative_import":
            prefix = next((c for c in module_node.children if c.type == "import_prefix"), None)
            dotted = next((c for c in module_node.children if c.type == "dotted_name"), None)
            level, module = len(self._text(prefix)), self._text(dotted)
        resolved = self._absolute_module(module, level)

        if any(c.type == "wildcard_import" for c in node.children):
            self._add_import(module, "*", None, level, resolved, line)
            return
        for item in node.children_by_field_name("name"):
            name, alias = self._name_and_alias(item)
            self._add_import(module, name, alias, level, resolved, line)
            scope.bind(alias or name, (resolved, name) if resolved else None)

    def _call(self, node: Node, scope: _Scope) -> None:
        func = node.child_by_field_name("function")
        if func.type == "identifier":
            self.raw_refs.append(("CALLS", scope.symbol, self._text(func), _line(node), scope))
        elif func.type == "attribute" and scope.self_name:
            obj = func.child_by_field_name("object")
            if obj.type == "identifier" and self._text(obj) == scope.self_name:
                name = f"{scope.self_name}.{self._text(func.child_by_field_name('attribute'))}"
                self.raw_refs.append(("CALLS", scope.symbol, name, _line(node), scope))

    # --- resolution --------------------------------------------------------------------

    def _resolve(self, relation: str, source: Symbol, name: str, line: int, scope: _Scope):
        value: Binding = None
        if "." in name:
            # self.method(): only methods defined in the enclosing class body (no MRO lookup).
            _, value = scope.parent.lookup(name.split(".", 1)[1])
            if not isinstance(value, str):
                value = None
        else:
            current: _Scope | None = scope
            while current is not None:
                # Class bodies are not visible from nested scopes (Python semantics).
                if current is scope or current.kind != "class":
                    found, value = current.lookup(name)
                    if found:
                        break
                current = current.parent

        return Reference(
            relation=relation,
            source_id=source.id,
            name=name,
            line=line,
            target_id=value if isinstance(value, str) else None,
            via_import=value if isinstance(value, tuple) else None,
        )

    # --- helpers -----------------------------------------------------------------------

    def _bind_targets(self, node: Node | None, scope: _Scope) -> None:
        if node is None:
            return
        if node.type == "identifier":
            scope.bind(self._text(node))
        for child in node.named_children:
            self._bind_targets(child, scope)

    def _parameter_names(self, params: Node | None) -> list[str]:
        names = []
        for p in params.named_children if params else []:
            if p.type == "identifier":
                names.append(self._text(p))
            elif p.type in {"default_parameter", "typed_default_parameter"}:
                names.append(self._text(p.child_by_field_name("name")))
            elif p.type in {"typed_parameter", "list_splat_pattern", "dictionary_splat_pattern"}:
                ident = next((c for c in p.named_children if c.type == "identifier"), None)
                if ident is not None:
                    names.append(self._text(ident))
        return names

    def _docstring(self, node: Node) -> str | None:
        body = node.child_by_field_name("body")
        first = body.named_children[0] if body and body.named_children else None
        if first is None or first.type != "expression_statement":
            return None
        string = first.named_children[0] if first.named_children else None
        if string is None or string.type != "string":
            return None
        parts = [self._text(c) for c in string.named_children if c.type == "string_content"]
        content = "".join(parts)
        return inspect.cleandoc(content) or None

    def _name_and_alias(self, item: Node) -> tuple[str, str | None]:
        if item.type == "aliased_import":
            name = self._text(item.child_by_field_name("name"))
            return name, self._text(item.child_by_field_name("alias"))
        return self._text(item), None

    def _add_import(self, module, name, alias, level, resolved, line) -> None:
        self.imports.append(Import(self.path, module, name, alias, level, resolved, line))

    def _absolute_module(self, module: str, level: int) -> str | None:
        if level == 0:
            return module
        package = list(PurePosixPath(self.path).parent.parts)
        if level - 1 > len(package):
            return None
        parts = package[: len(package) - (level - 1)] + (module.split(".") if module else [])
        return ".".join(parts) or None

    def _unique_id(self, base: str, node: Node) -> str:
        # Same qualified name twice (property setters, conditional defs): suffix the line.
        symbol_id = base if base not in self.ids else f"{base}@L{node.start_point[0] + 1}"
        self.ids.add(symbol_id)
        return symbol_id

    def _text(self, node: Node | None) -> str:
        if node is None:
            return ""
        return self.src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _line(node: Node) -> int:
    return node.start_point[0] + 1
