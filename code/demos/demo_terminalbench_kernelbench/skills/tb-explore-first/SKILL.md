---
name: tb-explore-first
description: A 1-command survey of the container (files, tools, services, versions) so you act on facts instead of guesses.
families: [terminalbench]
---
# Survey before you change anything (1 turn)

Run this as ONE bash call and read it fully:

```bash
pwd; ls -la; find . -maxdepth 2 -not -path '*/.*' | head -50
cat README* *.md 2>/dev/null | head -40
cat /etc/os-release | head -2; id
for t in python3 pip node npm gcc make cmake git curl jq sqlite3 psql redis-cli; do command -v $t >/dev/null && echo "have $t"; done
python3 --version; pip list 2>/dev/null | head -30
ps aux | grep -v ' \[' | head -15
```

Then re-read the task and note:
- every file path, port, name and format the task mentions. The hidden tests check those **exactly**.
- what already exists: an existing script, config or data file is usually meant to be edited, not replaced.
- for multi-container tasks: other services are reachable **by their service name**, not localhost.
