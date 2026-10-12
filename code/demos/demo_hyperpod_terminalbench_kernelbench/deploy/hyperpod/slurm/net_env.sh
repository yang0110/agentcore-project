#!/bin/bash
# Sourced INSIDE the trainer container on a GPU node (ray_node.sh, nccl_bw.sbatch): this node's VPC address
# and interface, and the NCCL transport between the nodes.
#   HP_EFA=1 (default)  NCCL over EFA (aws-ofi-nccl + libfabric's efa provider, Dockerfile.efa). Fails here if
#                       the node has no usable EFA device, rather than NCCL quietly falling back to sockets.
#   HP_EFA=0            NCCL over TCP sockets (the A/B baseline; also the only choice with the plain image).
# Sets NODE_IP, NODE_IFACE and exports the NCCL / libfabric env; Ray workers inherit it from `ray start`.
NODE_IP="$(hostname -I | tr ' ' '\n' | grep -m1 -F "${HP_SUBNET_PREFIX:-172.31.}")"
[ -n "$NODE_IP" ] || { echo "!! no ${HP_SUBNET_PREFIX:-172.31.}* address on $(hostname): $(hostname -I)" >&2; exit 1; }
# The interface holding NODE_IP (the image has no iproute2, so ask the kernel: SIOCGIFADDR per interface).
NODE_IFACE="$(python3 - "$NODE_IP" <<'EOF'
import fcntl, os, socket, struct, sys
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for name in sorted(os.listdir("/sys/class/net")):
    try:
        addr = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", name[:15].encode()))[20:24])
    except OSError:
        continue
    if addr == sys.argv[1]:
        print(name)
        break
EOF
)"
[ -n "$NODE_IFACE" ] || { echo "!! no interface holds $NODE_IP on $(hostname)" >&2; exit 1; }
# Bootstrap / out-of-band traffic (NCCL's, Gloo's) stays on the VPC interface either way.
export NCCL_SOCKET_IFNAME="$NODE_IFACE" GLOO_SOCKET_IFNAME="$NODE_IFACE"
if [ "${HP_EFA:-1}" = 1 ]; then
  command -v fi_info >/dev/null || { echo "!! HP_EFA=1 but this image has no libfabric (build it with Dockerfile.efa: hyperpod.py image)" >&2; exit 1; }
  N_EFA="$(fi_info -p efa -t FI_EP_RDM 2>/dev/null | grep -c '^provider: efa')"
  [ "$N_EFA" -gt 0 ] || { echo "!! HP_EFA=1 but fi_info finds no EFA device on $(hostname) (ls /dev/infiniband: $(ls /dev/infiniband 2>&1 | tr '\n' ' '))" >&2; exit 1; }
  # g6e (L40S) has no GPUDirect RDMA: aws-ofi-nccl stages through host memory, so FI_EFA_USE_DEVICE_RDMA stays unset.
  # No huge pages for libfabric's bounce buffers inside a container (Ray forks workers; huge pages + fork misbehave).
  export FI_PROVIDER=efa NCCL_NET_PLUGIN=ofi FI_EFA_USE_HUGE_PAGE=0
  NET_DESC="EFA (${N_EFA} endpoint(s), aws-ofi-nccl)"
else
  export NCCL_NET=Socket
  NET_DESC="TCP sockets on $NODE_IFACE"
fi
