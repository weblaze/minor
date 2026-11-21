"""Job runner executed inside a Kaggle script kernel.

launch.py prepends one line, `JOB = {...}`, to this file and pushes the result.
The runner clones the repository at the commit named in JOB, installs it,
resolves the named inputs under /kaggle/input, and runs the commands in order.
It stops at the first command that fails.

Written to /kaggle/working:
  job.json   the job, the resolved inputs, and return code and duration of each command
  logs/      combined stdout and stderr of each command
  runs/      run directories created by the commands (the repository's runs/ is a link to it)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = Path("/tmp/minor")
MAX_DEPTH = 6


def find_input(name):
    """Shallowest file or directory called `name` under /kaggle/input."""
    best = None
    for root, dirs, files in os.walk(INPUT):
        depth = len(Path(root).relative_to(INPUT).parts)
        if name in dirs or name in files:
            if best is None or depth < best[0]:
                best = (depth, Path(root) / name)
        if depth >= MAX_DEPTH or (best is not None and depth >= best[0]):
            dirs[:] = []
    if best is None:
        raise SystemExit(f"input {name!r} not found under {INPUT}")
    return str(best[1])


def run(command, log_path):
    start = time.time()
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, shell=True, cwd=REPO, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace")
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    return code, round(time.time() - start, 1)


def main():
    job = JOB  # noqa: F821, defined by the line launch.py prepends
    os.environ["PYTHONUNBUFFERED"] = "1"
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    (WORK / "runs").mkdir(exist_ok=True)

    subprocess.run(["git", "clone", "--quiet", job["repository"], str(REPO)], check=True)
    subprocess.run(["git", "checkout", "--quiet", job["commit"]], cwd=REPO, check=True)
    os.symlink(WORK / "runs", REPO / "runs")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", "."], cwd=REPO, check=True)

    inputs = {key: find_input(name) for key, name in job.get("inputs", {}).items()}
    record = {"job": job, "inputs": inputs, "commands": []}
    failed = False
    for index, template in enumerate(job["commands"]):
        command = template.format(**inputs)
        print(f"[job] {index}: {command}", flush=True)
        code, seconds = run(command, WORK / "logs" / f"{index:02d}.log")
        record["commands"].append({"command": command, "returncode": code, "seconds": seconds})
        with open(WORK / "job.json", "w") as f:
            json.dump(record, f, indent=2)
        if code != 0:
            failed = True
            break
    print(json.dumps(record["commands"], indent=2))
    if failed:
        raise SystemExit(1)


main()
