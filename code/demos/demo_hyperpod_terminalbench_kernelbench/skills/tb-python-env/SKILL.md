---
name: tb-python-env
description: Installing Python packages and running scripts reliably in task containers (which python, pip flags, offline/slow installs).
families: [terminalbench]
---
# Python environment

- Use the interpreter the tests will use: `python3 -m pip install …`, never a bare `pip` (it may belong to
  another Python). Check with `python3 -c "import sys;print(sys.executable)"`.
- If pip refuses with "externally-managed-environment", add `--break-system-packages`.
- Pin only what the task asks for. Quiet + bounded: `timeout 170 python3 -m pip install -q <pkg>==<ver>`.
- No network? `pip download` won't work. Look for local wheels (`find / -name '*.whl' 2>/dev/null | head`)
  and `pip install --no-index --find-links <dir> <pkg>`.
- Already installed? `python3 -c "import <mod>, sys; print(<mod>.__version__)"` before installing.
- A script must run with exactly the command the task gives. Test it that way, from the directory
  the task implies, and read the full traceback (the last line is the error; the first frame inside your
  file is the cause).
- Data tasks: check the output format (CSV header, JSON keys, dtypes) by printing the first lines of what
  you wrote, e.g. `head -3 out.csv`, or `python3 -c "import json;print(json.load(open('out.json')))"`.
