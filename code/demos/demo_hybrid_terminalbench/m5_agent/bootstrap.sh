#!/bin/bash
# cloud-init user-data for the m5 Docker host (Amazon Linux 2023, x86_64). deploy/m5_host.py fills the
# __PLACEHOLDERS__ and gzips this (EC2 user-data is capped at 16 KB). Installs Docker + the compose and
# buildx plugins, locks the containers away from the host and from instance metadata, and runs
# compose_agent.py as a systemd unit on :__AGENT_PORT__.
set -euxo pipefail
exec > >(tee -a /var/log/acs-bootstrap.log) 2>&1

dnf install -y docker iptables-nft python3
ARCH=x86_64
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL "https://github.com/docker/compose/releases/download/__COMPOSE_VERSION__/docker-compose-linux-$ARCH" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
curl -fsSL "https://github.com/docker/buildx/releases/download/__BUILDX_VERSION__/buildx-__BUILDX_VERSION__.linux-amd64" \
  -o /usr/local/lib/docker/cli-plugins/docker-buildx
chmod +x /usr/local/lib/docker/cli-plugins/docker-*

# Images + build cache live on the big root volume; cap container logs.
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<'JSON'
{"log-driver": "json-file", "log-opts": {"max-size": "20m", "max-file": "2"},
 "default-address-pools": [{"base": "10.200.0.0/16", "size": 24}]}
JSON
systemctl enable --now docker

# ---- isolation rules (re-applied by the agent unit on every start: docker rebuilds its chains) ----
cat > /usr/local/sbin/acs-firewall <<'FW'
#!/bin/bash
# Containers run policy-model code as root. They may reach the internet (NAT) and the trainer's
# adapter, but never this host (the agent API, sshd, the docker daemon) nor instance metadata.
set -e
iptables -N ACS-IN 2>/dev/null || iptables -F ACS-IN
iptables -A ACS-IN -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A ACS-IN -j DROP
for IF in docker0 br-+; do
  iptables -C INPUT -i "$IF" -j ACS-IN 2>/dev/null || iptables -I INPUT -i "$IF" -j ACS-IN
done
iptables -N DOCKER-USER 2>/dev/null || true
for DST in 169.254.169.254 169.254.170.2; do
  iptables -C DOCKER-USER -d "$DST" -j DROP 2>/dev/null || iptables -I DOCKER-USER -d "$DST" -j DROP
done
FW
chmod +x /usr/local/sbin/acs-firewall
/usr/local/sbin/acs-firewall

# ---- the agent ----
install -d -m 700 /srv/acs
umask 077
echo '__TOKEN__' > /srv/acs/token
base64 -d > /srv/acs/compose_agent.py <<'B64'
__AGENT_B64__
B64
cat > /etc/systemd/system/acs-compose-agent.service <<'UNIT'
[Unit]
Description=demo_hybrid_terminalbench compose agent
After=docker.service network-online.target
Requires=docker.service

[Service]
Environment=ACS_ROOT=/srv/acs ACS_PORT=__AGENT_PORT__ ACS_MAX_PROJECTS=__MAX_PROJECTS__ ACS_CLIENT_MEM=__CLIENT_MEM__ ACS_CLIENT_CPUS=__CLIENT_CPUS__
ExecStartPre=/usr/local/sbin/acs-firewall
ExecStart=/usr/bin/python3 /srv/acs/compose_agent.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now acs-compose-agent
echo "acs bootstrap done"
