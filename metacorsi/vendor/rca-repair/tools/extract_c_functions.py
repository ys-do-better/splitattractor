#!/usr/bin/env python3
"""Copy named function definitions (and named top-level declarations) verbatim out of a C file.

usage: extract_c_functions.py SRC NAME [NAME ...] > OUT

Used to build the example targets from upstream sources without retyping them: each function keeps its exact text,
so line-level evidence (crash lines, guards, patch sites) refers to upstream code.
"""
import sys

import tree_sitter_c as tsc
from tree_sitter import Language, Parser

C = Language(tsc.language())


def func_name(node):
    d = node.child_by_field_name("declarator")
    while d is not None and d.type != "identifier":
        d = d.child_by_field_name("declarator")
    return d.text.decode() if d is not None else None


def main():
    src_path, names = sys.argv[1], sys.argv[2:]
    src = open(src_path, "rb").read()
    tree = Parser(C).parse(src)
    found = {}
    for node in tree.root_node.children:
        if node.type == "function_definition":
            n = func_name(node)
            if n in names:
                found[n] = src[node.start_byte:node.end_byte].decode()
    missing = [n for n in names if n not in found]
    if missing:
        sys.exit(f"not found in {src_path}: {', '.join(missing)}")
    for n in names:  # keep the requested order
        sys.stdout.write(found[n] + "\n\n")


if __name__ == "__main__":
    main()
