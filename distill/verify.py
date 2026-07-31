"""Verification for reasoning traces: is the final answer actually correct?

Rejection sampling needs this — generate N traces, keep only verified ones.
Findings this implements (OpenThoughts / ICLR 2026):
  - 1 -> 16 samples per question moved peak 75.16 -> 85.01, so sample MANY
  - quality > quantity (114K verified beat 800K unverified)
  - NEVER strip self-reflection from traces (-49.1% relative if removed)

ponytail: sympy for math, subprocess-sandboxed exec for code. No LLM judge —
unverifiable domains are out of scope until the verifiable ones are working.
"""

import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S)
FINAL = re.compile(r"(?:final answer|answer)\s*(?:is)?\s*[:=]?\s*\$?(-?[\d,./]+)", re.I)


@dataclass
class Result:
    ok: bool
    extracted: str = ""
    reason: str = ""


def extract_answer(trace: str) -> str:
    """Last \\boxed{} wins, then <answer>, then a trailing 'answer is X'."""
    for pat in (BOXED, ANSWER_TAG, FINAL):
        hits = pat.findall(trace)
        if hits:
            return hits[-1].strip()
    return ""


def _norm(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\(?:text|mathrm|mbox)\{([^}]*)\}", r"\1", s)
    # escaped forms BEFORE bare ones, else "\$175" -> "\175" leaves a stray backslash
    for tok in ("\\$", "\\%", "\\left", "\\right", "\\!", "\\,", "\\;", "\\:"):
        s = s.replace(tok, "")
    s = s.replace(" ", "").replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    return s.lower()


def _latex_to_sympy(s: str) -> str:
    """Enough LaTeX to cover competition answers. sympy's parse_latex needs
    antlr (not installed) so we normalize by hand instead."""
    s = _norm(s)
    for _ in range(3):  # nested fractions
        s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    s = s.replace("\\pi", "pi").replace("^", "**").replace("{", "(").replace("}", ")")
    return s.strip()


def check_math(trace: str, gold: str, timeout_s: int = 5) -> Result:
    got = extract_answer(trace)
    if not got:
        return Result(False, "", "no answer found")
    if _norm(got) == _norm(gold):
        return Result(True, got, "exact")
    # symbolic equality: 1/2 == 0.5 == 2/4 == \frac{1}{2}
    try:
        from sympy import nsimplify, simplify
        from sympy.parsing.sympy_parser import parse_expr

        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(timeout_s)
        try:
            a, b = (parse_expr(_latex_to_sympy(x), evaluate=True) for x in (got, gold))
            eq = simplify(a - b) == 0
            if not eq:  # numeric fallback for decimals vs exact forms
                eq = abs(float(nsimplify(a) - nsimplify(b))) < 1e-6
        finally:
            signal.alarm(0)
        return Result(bool(eq), got, "symbolic" if eq else "mismatch")
    except Exception as e:
        return Result(False, got, f"unparsed: {type(e).__name__}")


def check_code(trace: str, tests: str, timeout_s: int = 10) -> Result:
    """Run extracted code + tests in a subprocess. NOT a security sandbox —
    ponytail: fine for our own generated traces, needs firejail/docker if the
    source is ever untrusted."""
    blocks = re.findall(r"```(?:python)?\n(.*?)```", trace, re.S)
    if not blocks:
        return Result(False, "", "no code block")
    code = blocks[-1]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + tests)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout_s)
        return Result(p.returncode == 0, code[:80], p.stderr.decode()[-200:] if p.returncode else "pass")
    except subprocess.TimeoutExpired:
        return Result(False, code[:80], "timeout")


def verify(trace: str, item: dict) -> Result:
    kind = item.get("kind", "math")
    if kind == "math":
        return check_math(trace, str(item["answer"]))
    if kind == "code":
        return check_code(trace, item["tests"])
    return Result(False, "", f"unknown kind {kind}")


def demo():
    assert extract_answer(r"so \boxed{42}") == "42"
    assert extract_answer("blah <answer>7</answer>") == "7"
    assert check_math(r"...\boxed{1/2}", "0.5").ok, "symbolic equality failed"
    assert check_math(r"...\boxed{\frac{1}{10}}", "0.1").ok, "latex frac failed"
    assert check_math(r"...\boxed{\$175}", "175").ok, "dollar-sign answer failed"
    assert check_math(r"...\boxed{50\%}", "50").ok, "percent answer failed"
    assert check_math(r"...\boxed{\dfrac{3}{4}}", r"\frac{6}{8}").ok, "frac-vs-frac failed"
    assert not check_math(r"...\boxed{\frac{1}{10}}", "0.2").ok, "false positive"
    assert check_math(r"...\boxed{42}", "42").ok
    assert not check_math(r"...\boxed{41}", "42").ok
    assert not check_math("no answer here", "42").ok
    assert check_code("```python\ndef f(x): return x*2\n```", "assert f(2)==4").ok
    assert not check_code("```python\ndef f(x): return x*3\n```", "assert f(2)==4").ok
    print("verify: all checks pass")


if __name__ == "__main__":
    demo()
