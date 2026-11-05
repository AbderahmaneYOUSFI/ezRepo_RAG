from pathlib import Path

import pytest

from ez_repo.ingest import SourceFile, collect_python_files
from ez_repo.parser import module_name, parse_file, parse_repository

DEMO_REPO = Path(__file__).parent / "fixtures" / "demo_repo"


@pytest.fixture(scope="module")
def graph():
    files = collect_python_files(DEMO_REPO, 10**6, 100, 10**7).files
    return parse_repository(files)


@pytest.fixture(scope="module")
def symbols(graph):
    return {s.id: s for s in graph.symbols}


def edges(graph, rel_type):
    return {(r.source_id, r.target_id) for r in graph.relations if r.type == rel_type}


@pytest.mark.parametrize(
    ("path", "expected"),
    [("pkg/core.py", "pkg.core"), ("pkg/__init__.py", "pkg"), ("app.py", "app")],
)
def test_module_name(path, expected):
    assert module_name(path) == expected


def test_extracts_classes_functions_methods(symbols):
    kinds = {sid: s.kind for sid, s in symbols.items() if sid.startswith("pkg/core.py")}
    assert kinds["pkg/core.py::greet"] == "function"
    assert kinds["pkg/core.py::Greeter"] == "class"
    assert kinds["pkg/core.py::Greeter.run"] == "method"
    assert kinds["pkg/core.py::outer.inner"] == "function"


def test_symbol_metadata(symbols):
    greet = symbols["pkg/core.py::greet"]
    assert greet.path == "pkg/core.py"
    assert greet.signature == "def greet(name: str) -> str"
    assert greet.docstring == "Return a greeting.\n\nUses the slug of the name."
    assert greet.source.startswith("def greet(")
    assert greet.end_line - greet.start_line == greet.source.count("\n")
    assert symbols["pkg/core.py::outer"].signature == "async def outer()"

    greeter = symbols["pkg/core.py::Greeter"]
    assert greeter.bases == ("Base",)
    assert greeter.docstring == "Greets people."
    assert symbols["pkg/core.py::Greeter.run"].parent_id == greeter.id


def test_decorators_are_kept_in_source_and_lines(symbols):
    label = symbols["pkg/core.py::Greeter.label"]
    assert label.decorators == ("property",)
    assert label.source.startswith("@property")


def test_duplicate_names_get_unique_ids(symbols):
    labels = [s for s in symbols.values() if s.qualified_name == "Greeter.label"]
    assert len(labels) == 2
    assert len({s.id for s in labels}) == 2


def test_imports(graph):
    core = {(i.module, i.name, i.alias, i.level, i.resolved_module) for i in graph.imports
            if i.path == "pkg/core.py"}  # fmt: skip
    assert ("json", None, "js", 0, "json") in core
    assert ("utils", "slugify", "make_slug", 1, "pkg.utils") in core
    assert ("", "utils", None, 1, "pkg") in core


def test_defines_relations(graph):
    defines = edges(graph, "DEFINES")
    assert ("pkg/core.py", "pkg/core.py::Greeter") in defines
    assert ("pkg/core.py::Greeter", "pkg/core.py::Greeter.run") in defines
    assert ("pkg/core.py::outer", "pkg/core.py::outer.inner") in defines


def test_imports_relations(graph):
    imports = edges(graph, "IMPORTS")
    assert ("app.py", "pkg/__init__.py") in imports
    assert ("pkg/core.py", "pkg/utils.py") in imports  # "from . import utils" -> submodule
    assert ("pkg/core.py", "module:os") in imports
    assert graph.external_modules == ["json", "os", "requests", "typing"]


def test_resolved_calls(graph):
    calls = edges(graph, "CALLS")
    assert ("pkg/core.py::greet", "pkg/core.py::format_message") in calls  # same file
    assert ("pkg/core.py::greet", "pkg/utils.py::slugify") in calls  # aliased import
    assert ("pkg/core.py::Greeter.run", "pkg/core.py::Greeter.reset") in calls  # self.method()
    assert ("pkg/core.py::outer", "pkg/core.py::outer.inner") in calls  # closure
    assert ("app.py::App.main", "pkg/core.py::greet") in calls  # re-exported by pkg/__init__


def test_unresolved_calls_produce_no_relation(graph):
    callers = edges(graph, "CALLS")
    sources = {src for src, _ in callers}
    assert "pkg/core.py::shadowed" not in sources  # parameter shadows the function
    assert "pkg/core.py::Greeter.build" not in sources  # staticmethod: no self
    # Inherited method (no MRO resolution) and builtins are not linked.
    run_targets = {dst for src, dst in callers if src == "pkg/core.py::Greeter.run"}
    assert run_targets == {"pkg/core.py::Greeter.reset", "pkg/core.py::greet"}
    assert all(dst.split("::")[0] in {"app.py", "pkg/core.py", "pkg/utils.py"}
               for _, dst in callers)  # fmt: skip


def test_inherits_relations(graph):
    assert edges(graph, "INHERITS") == {
        ("pkg/core.py::Greeter", "pkg/base.py::Base"),
        ("app.py::App", "pkg/core.py::Greeter"),
    }


def test_every_relation_points_to_known_nodes(graph, symbols):
    files = {f.path for f in graph.files}
    for rel in graph.relations:
        assert rel.source_id in symbols or rel.source_id in files
        assert (
            rel.target_id in symbols
            or rel.target_id in files
            or rel.target_id.startswith("module:")
        )


def test_syntax_errors_are_tolerated(graph):
    broken = next(f for f in graph.files if f.path == "broken.py")
    assert broken.has_errors
    assert "broken.py::ok" in {s.id for s in broken.symbols}


def test_empty_and_comment_only_files():
    for text in ["", "# just a comment\n", "\n\n"]:
        parsed = parse_file("empty.py", text)
        assert parsed.symbols == [] and parsed.imports == [] and not parsed.has_errors


def test_rebound_name_is_not_resolved():
    source = "def helper():\n    pass\n\nhelper = wrap(helper)\n\ndef main():\n    helper()\n"
    graph = parse_repository_from({"m.py": source})
    assert not edges(graph, "CALLS")


def test_relative_import_beyond_root_is_ignored():
    graph = parse_repository_from({"top.py": "from .. import nothing\n"})
    assert graph.imports[0].resolved_module is None
    assert not edges(graph, "IMPORTS")


def test_unicode_source_keeps_correct_text():
    parsed = parse_file("u.py", 'def café():\n    """Résumé ✓"""\n')
    assert parsed.symbols[0].name == "café"
    assert parsed.symbols[0].docstring == "Résumé ✓"


def parse_repository_from(sources: dict[str, str]):
    return parse_repository(SourceFile(p, t, len(t)) for p, t in sources.items())
