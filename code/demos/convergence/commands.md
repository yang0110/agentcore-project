# Convergence commands

Run this inside the container. Only one run at a time: a second Ray start kills the first job.

```bash
docker exec -it agentcore_p4 bash
cd /work/code/demos/convergence

# Terminal-Bench: pool of ~218 task Runtimes
bash 20_cpu_terminalbench.sh
```

To keep the run going after the terminal closes, start it detached:

```bash
docker exec -d agentcore_p4 bash -lc 'cd /work/code/demos/convergence && bash 20_cpu_terminalbench.sh > /tmp/conv20.out 2>&1'
```
