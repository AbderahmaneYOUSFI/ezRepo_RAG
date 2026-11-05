"""Checks that the Tree-sitter Python grammar is installed and usable."""

import tree_sitter_python
from tree_sitter import Language, Parser


def test_parse_python_source():
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(b"import os\n\nclass A:\n    def f(self):\n        pass\n")
    kinds = [node.type for node in tree.root_node.children]
    assert kinds == ["import_statement", "class_definition"]
