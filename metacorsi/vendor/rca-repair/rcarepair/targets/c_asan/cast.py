"""Small C syntax helpers on tree-sitter: functions, calls, guards, loops, error exits.

Line numbers are 1-based, as in compiler and sanitizer output.
"""
import re
from dataclasses import dataclass, field

import tree_sitter_c as tsc
from tree_sitter import Language, Parser

_C = Language(tsc.language())
_PARSER = Parser(_C)
CMP_OPS = {"<", "<=", ">", ">=", "==", "!="}
STMT_TYPES = {"expression_statement", "declaration", "return_statement", "if_statement", "for_statement",
              "while_statement", "do_statement", "goto_statement", "switch_statement", "compound_statement",
              "break_statement", "continue_statement"}


def norm(expr):
    """compare expressions modulo whitespace and redundant outer parentheses"""
    e = re.sub(r"\s+", "", expr)
    while e.startswith("(") and e.endswith(")") and _balanced(e[1:-1]):
        e = e[1:-1]
    return e


def _balanced(s):
    d = 0
    for ch in s:
        d += ch == "("
        d -= ch == ")"
        if d < 0:
            return False
    return d == 0


def text(node):
    return node.text.decode()


def line_of(node):
    return node.start_point[0] + 1


def end_line_of(node):
    return node.end_point[0] + 1


def walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def _decl_name(d):
    while d is not None and d.type not in ("identifier", "field_identifier"):
        nxt = d.child_by_field_name("declarator")
        if nxt is None:
            ids = [c for c in d.children if c.type == "identifier"]
            return text(ids[0]) if ids else None
        d = nxt
    return text(d) if d is not None else None


@dataclass
class Function:
    name: str
    node: object
    params: list
    start: int
    end: int


@dataclass
class Call:
    callee: str
    args: list
    line: int
    end_line: int
    node: object = field(repr=False)


@dataclass
class Guard:
    cond: str
    line: int
    comparisons: list          # [(lhs, op, rhs)] normalised
    exits: bool                # then-branch leaves (return / goto / break / continue)
    then_text: str
    node: object = field(repr=False)


class CFile:
    def __init__(self, path):
        self.path = path
        self.src = open(path, "rb").read()
        self.lines = self.src.decode(errors="replace").split("\n")
        self.tree = _PARSER.parse(self.src)
        self.functions = {}
        for n in self.tree.root_node.children:
            if n.type == "function_definition":
                fd = n.child_by_field_name("declarator")
                while fd is not None and fd.type != "function_declarator":
                    fd = fd.child_by_field_name("declarator")
                if fd is None:
                    continue
                name = _decl_name(fd.child_by_field_name("declarator"))
                params = []
                plist = fd.child_by_field_name("parameters")
                for p in plist.children if plist else []:
                    if p.type == "parameter_declaration":
                        params.append(_decl_name(p.child_by_field_name("declarator")))
                self.functions[name] = Function(name, n, params, line_of(n), end_line_of(n))

    # ---- lookup -------------------------------------------------------------------------------------------
    def function_at(self, line):
        for f in self.functions.values():
            if f.start <= line <= f.end:
                return f
        return None

    def calls(self, fn):
        out = []
        for n in walk(fn.node):
            if n.type == "call_expression":
                callee = n.child_by_field_name("function")
                args = n.child_by_field_name("arguments")
                out.append(Call(text(callee), [text(a) for a in args.named_children if a.type != "comment"],
                                line_of(n), end_line_of(n), n))
        return out

    def call_at(self, line, callee=None):
        """the call covering `line` (sanitizer and gdb report any line of a multi-line call)"""
        fn = self.function_at(line)
        if fn is None:
            return None
        hits = [c for c in self.calls(fn) if c.line <= line <= c.end_line and (callee is None or c.callee == callee)]
        hits.sort(key=lambda c: (c.end_line - c.line, -c.line))
        return hits[0] if hits else None

    def statement_of(self, node):
        n = node
        while n is not None and n.type not in STMT_TYPES:
            n = n.parent
        return n

    def statement_at(self, line):
        """innermost simple statement that starts on `line`"""
        fn = self.function_at(line)
        best = None
        for n in walk(fn.node):
            if n.type in STMT_TYPES and n.type != "compound_statement" and line_of(n) == line:
                if best is None or (n.end_byte - n.start_byte) < (best.end_byte - best.start_byte):
                    best = n
        return best

    def enclosing(self, node, types):
        n = node.parent
        while n is not None and n.type not in types:
            n = n.parent
        return n

    def indent_of(self, line):
        s = self.lines[line - 1]
        return s[: len(s) - len(s.lstrip())]

    # ---- facts used by the RCA ------------------------------------------------------------------------------
    def store_target(self, stmt):
        """for `lhs = rhs;` return (lhs, rhs) texts, else None"""
        for n in walk(stmt):
            if n.type == "assignment_expression":
                return text(n.child_by_field_name("left")), text(n.child_by_field_name("right"))
        return None

    def subscript_index(self, lhs_node_text):
        m = re.match(r"^\s*([\w\.\->]+)\s*\[(.+)\]\s*$", lhs_node_text)
        return (m.group(1), m.group(2).strip()) if m else (None, None)

    def loop_bound(self, stmt, var):
        """in `for (...; var < X; ...)` around stmt, return X"""
        loop = self.enclosing(stmt, {"for_statement"})
        while loop is not None:
            cond = loop.child_by_field_name("condition")
            if cond is not None:
                for lhs, op, rhs in comparisons(cond):
                    if lhs == norm(var) and op in ("<", "<="):
                        return rhs, line_of(loop)
                    if rhs == norm(var) and op in (">", ">="):
                        return lhs, line_of(loop)
            loop = self.enclosing(loop, {"for_statement"})
        return None, None

    def guards(self, fn, before_line=None):
        out = []
        for n in walk(fn.node):
            if n.type != "if_statement":
                continue
            if before_line is not None and line_of(n) >= before_line:
                continue
            cond = n.child_by_field_name("condition")
            then = n.child_by_field_name("consequence")
            exits = any(m.type in ("return_statement", "goto_statement", "break_statement", "continue_statement")
                        for m in walk(then))
            out.append(Guard(text(cond), line_of(n), comparisons(cond), exits, text(then), n))
        return out

    def guards_after(self, fn, after_line):
        return [g for g in self.guards(fn) if g.line > after_line]

    def error_exits(self, fn):
        """if-branches that set an error code and leave: [(assign_lhs, errno, exit_stmt_text, line)]"""
        out = []
        for g in self.guards(fn):
            then = g.node.child_by_field_name("consequence")
            assign = None
            leave = None
            for m in walk(then):
                if m.type == "assignment_expression" and re.match(r"^-\s*E[A-Z0-9]+$", text(m.child_by_field_name("right")).strip()):
                    assign = (text(m.child_by_field_name("left")).strip(), text(m.child_by_field_name("right")).strip())
                if m.type in ("goto_statement", "return_statement") and leave is None:
                    leave = text(m).strip()
            if assign and leave:
                out.append((assign[0], assign[1], leave, g.line))
        return out


def comparisons(node):
    out = []
    for n in walk(node):
        if n.type == "binary_expression":
            op = n.child_by_field_name("operator")
            if op is not None and text(op) in CMP_OPS:
                out.append((norm(text(n.child_by_field_name("left"))), text(op),
                            norm(text(n.child_by_field_name("right")))))
    return out


def upper_bounds(comps, expr):
    """comparisons that put an upper bound on expr (expr > X, expr >= X, X < expr, X <= expr, and the <= forms)"""
    e = norm(expr)
    out = []
    for lhs, op, rhs in comps:
        if lhs == e and op in (">", ">=", "<", "<="):
            out.append((lhs, op, rhs))
        elif rhs == e and op in (">", ">=", "<", "<="):
            out.append((lhs, op, rhs))
    return [c for c in out if not _is_zero(c[0] if c[0] != e else c[2])]


def _is_zero(x):
    return x in ("0", "0U", "0u")


def resolve_define(include_dirs, name):
    """value of `#define name <int>` in the headers"""
    import glob
    import os
    for d in include_dirs:
        for p in glob.glob(os.path.join(d, "**", "*.h"), recursive=True):
            for ln in open(p, errors="replace"):
                m = re.match(r"\s*#\s*define\s+%s\s+\(?\s*(\d+)\s*\)?\s*(/\*.*)?$" % re.escape(name), ln)
                if m:
                    return int(m.group(1))
    return None
