"""Front door: run one stage of one method.

  python scripts/run.py <method> <stage> [args...]   runs methods/<method>/<stage>.py with args
  python scripts/run.py --list                        lists methods and stages

A stage is any script in methods/<method>/ except the shared modules common.py and data.py.
The stage runs in this process exactly as if it had been started directly
(`python methods/<method>/<stage>.py args...`), which keeps working.
"""
import ast
import runpy
import sys
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1] / "methods"
MODULES = {"common", "data"}


def stages(method):
    return sorted(p.stem for p in (METHODS / method).glob("*.py") if p.stem not in MODULES)


def methods():
    return sorted(p.name for p in METHODS.iterdir() if p.is_dir() and any(p.glob("*.py")))


def summary(path):
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    return doc.splitlines()[0] if doc else ""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    if sys.argv[1] == "--list":
        for method in methods():
            print(method)
            for stage in stages(method):
                print(f"  {stage:16s} {summary(METHODS / method / (stage + '.py'))}")
        return
    if len(sys.argv) < 3:
        raise SystemExit("usage: python scripts/run.py <method> <stage> [args...]; --list shows both")
    method, stage = sys.argv[1], sys.argv[2]
    if method not in methods():
        raise SystemExit(f"unknown method {method!r}; available: {', '.join(methods())}")
    if stage not in stages(method):
        raise SystemExit(f"unknown stage {stage!r} of {method}; available: {', '.join(stages(method))}")
    path = METHODS / method / f"{stage}.py"
    sys.argv = [str(path), *sys.argv[3:]]
    sys.path.insert(0, str(path.parent))
    runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
