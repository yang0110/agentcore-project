---
name: tb-build-from-source
description: Triage for compile/build tasks (make, cmake, configure, gcc errors, missing headers/libs) in few turns.
families: [terminalbench]
---
# Building from source

1. Find the build system: `ls` for `CMakeLists.txt` / `configure` / `Makefile` / `setup.py` / `Cargo.toml`.
2. Build bounded and keep the log: `timeout 170 make -j$(nproc) > /tmp/build.log 2>&1; echo exit=$?`
3. Read the **first** error, not the last: `grep -n -m5 -E 'error|Error' /tmp/build.log`.
   Later errors are usually consequences of the first.
4. Map the error to a fix:

| Error | Fix |
|---|---|
| `fatal error: foo.h: No such file` | the `-dev` package: `apt-get install -y libfoo-dev` (search: `apt-cache search foo`) |
| `undefined reference to 'bar'` | a missing `-l<lib>` at link time, or link order (libs after objects) |
| `command not found: cmake/make/gcc` | `apt-get update -qq && apt-get install -y -qq build-essential cmake` |
| cmake can't find a package | `-D<Pkg>_DIR=…`, or install `<pkg>-dev` |
| warnings treated as errors | remove `-Werror` from the flags, don't edit the code to silence them |

5. Install where the task says (`make install PREFIX=…`). Then run the produced binary once with the
   example from the task to prove it works.
