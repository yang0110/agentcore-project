# Running docker-compose tasks on AWS (outside AgentCore)

**Scope.** AgentCore can't run `docker compose` on any substrate: every capability is dropped,
`NoNewPrivs=1`, seccomp is on, and there is no `/dev/kvm`. See `terminalbench-unsupported-tasks.md` §2.
A plain EC2 instance (m5 or any other type) is a full VM with root and a real kernel, so Docker and
Compose run there as they do on a laptop. This doc lists three ways to run them on AWS.

## 1. Plain EC2 with Docker (simplest; the closest match to how Terminal-Bench runs upstream)

Install Docker and the compose plugin.

**Amazon Linux 2023:**

```bash
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user          # log out/in afterwards
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m) \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
docker compose version
```

**Ubuntu:**

```bash
curl -fsSL https://get.docker.com | sudo sh   # includes the compose plugin
sudo usermod -aG docker ubuntu
```

Then run a task:

```bash
cd terminalbench/tasks/postgres-csv-clean
docker compose -p ep_0001 up -d --build      # -p = unique project name per episode
docker compose -p ep_0001 exec client bash   # where the agent works
docker compose -p ep_0001 down -v            # tear down + delete volumes
```

**Running many episodes at once on one host.** Give each episode its own `-p <name>`. Each project
then gets its own network, containers and volumes, and the service names (`db`, `redis`) resolve only
inside that project. Ports don't collide, because the services talk over the project's own network
rather than host ports. Only `ports:` entries that publish to the host can clash; drop them, or let
Docker pick random host ports.

**Sizing:**

- You're limited by the host's CPU, RAM and disk. Plan roughly 0.5–1 vCPU and 1–2 GB per episode.
- `m5.4xlarge` (16 vCPU / 64 GB) handles about 16–30 episodes at once.
- Give it a big gp3 EBS volume, at least 200 GB, for the images.
- For more scale, run several hosts.

**Driving it from the trainer:**

- Point the Docker CLI at the instance: `DOCKER_HOST=ssh://ec2-user@<ip> docker compose ...`.
- Alternatively, run a small agent server on the instance and call it over the VPC.
- The Terminal-Bench harness (`tb run`) works like this natively: it calls `docker compose` itself.

## 2. ECS on EC2 or Fargate (managed, scales out, one task per episode)

ECS doesn't take a compose file directly, so you convert it once:

- Translate each compose service into a container in an ECS task definition.
- Containers in the same task share `localhost` (`awsvpc` mode). Replace hostnames like `db` with
  `127.0.0.1`, or add `extraHosts`.
- Tools such as **compose-x** (`ecs-compose-x`) or a small converter script do the translation.

Then start one task per episode with `aws ecs run-task`, and stop it with `aws ecs stop-task` when the
episode ends.

- **Pros:** no hosts to manage, strong isolation per episode (each task gets its own network
  interface), scales to hundreds of episodes.
- **Cons:** about 30–60 s to start each task, and you have to translate each compose file.

Fargate is x86 or arm64 and allows up to 200 GB of ephemeral storage, which covers the x86 and
big-image tasks too.

## 3. EKS (Kubernetes)

Convert with `kompose convert -f docker-compose.yaml` and run one namespace or pod per episode. This
only makes sense if you already run EKS; otherwise it's the most moving parts.

## Recommendation for the 23 unsupported Terminal-Bench tasks

Start with **option 1**: an x86 `m5.4xlarge` with Docker, one `docker compose -p` project per episode,
and a big EBS volume. It runs all 23 unchanged:

- **Compose tasks:** run natively.
- **x86-only tasks:** run natively on x86.
- **Big images:** fit on the large EBS volume.
- **qemu tasks:** run, but under software emulation (QEMU TCG), since the m5 has no KVM.
- **Exception:** the 3 tasks broken upstream still need their patches.

Move to **option 2 (ECS)** when one host can't run enough episodes at once.

**Security caveat.** The agent's commands run inside the task's client container. If that's an RL
policy model running arbitrary code, either:

- don't mount the Docker socket into it, and keep `--privileged` off, or
- use a dedicated instance with no IAM role, so a container escape can't reach your account.

## Related

- `docs/terminalbench-unsupported-tasks.md`: the 23 left-out tasks and the AgentCore-side options.
- `docs/agentcore-networking.md`: SG→SG rules if the trainer reaches the host over the VPC.
