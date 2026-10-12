---
name: tb-verify-before-stop
description: A final checklist to run before you stop; the task is graded by hidden tests that check exact paths, names, formats and running state.
families: [terminalbench]
---
# Before you stop (1 turn)

Hidden tests run after you stop. They cannot see your explanation, only the container's state.

1. Re-read the task. Make a list of every concrete requirement (file X at path Y, command Z prints W,
   service on port P, permission, exact output format).
2. Check each one with a command in ONE bash call, for example:
   ```bash
   ls -la /path/to/expected/file && head -5 /path/to/expected/file
   ./the_command_from_the_task arg1 ; echo exit=$?
   curl -s localhost:PORT/ | head -3
   ```
3. Fix anything that fails and re-check only that item.
4. Leave no temporary edits that change behaviour (debug prints in output files, test data you added
   to the data dir, a stopped service).

Common silent failures: the wrong filename or extension; output to stdout when a file was asked for (or the reverse);
a trailing newline or extra header in a format-checked file; a relative path where an absolute one was
asked for; a service started in the foreground and killed when the call timed out.
