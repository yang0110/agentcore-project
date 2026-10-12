---
name: tb-long-running
description: How to start servers/daemons and long jobs without hanging your bash tool, and how to wait until they are ready.
families: [terminalbench]
---
# Servers and long jobs

A bash call that never returns burns the whole episode. A foreground server never returns.

Start it detached, with its log in a file:
```bash
nohup python3 app.py > /tmp/app.log 2>&1 &
# or: setsid ./server --port 8000 > /tmp/server.log 2>&1 < /dev/null &
```

Wait until it's ready (bounded), then prove it:
```bash
for i in $(seq 30); do curl -fs localhost:8000/health >/dev/null && break; sleep 1; done
curl -s localhost:8000/ | head; tail -5 /tmp/app.log
```

- No curl? Use `python3 -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8000').status)"`.
- Check what is listening: `ss -ltnp 2>/dev/null || netstat -ltnp`.
- Long builds and downloads: wrap them in `timeout 170 <cmd>` so a stuck step returns with an error.
- The service must still be running **when you stop**: the tests run afterwards. Don't kill it, and don't
  start it inside a subshell that exits.
- Package-installed services: `service <name> start` (there is usually no systemd in a container).
