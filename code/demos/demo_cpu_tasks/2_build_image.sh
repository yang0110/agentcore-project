#!/bin/bash
# Step 2/6 — build this demo's sandbox image(s) from deploy/Dockerfile.* (HOST; build context = this dir).
# microVM Runtimes need linux/arm64 (cross-built via buildx + QEMU, <= 2048 MB); capacity-provider
# Instances (m5 / g6) need linux/amd64. IMAGES="name:Dockerfile ..." in deploy/demo.env (override to subset).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host

if [ -n "${BUILD_CMD:-}" ]; then log "step 2/6: $BUILD_CMD"; (cd "$HERE" && eval "$BUILD_CMD"); exit 0; fi

log "step 2/6: build $(image_names | tr '\n' ' ')for $PLATFORM ($SUBSTRATE)"
CROSS=0
if [ "$PLATFORM" = linux/arm64 ] && [ "$(uname -m)" != aarch64 ]; then
  CROSS=1
  docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null
  docker buildx inspect "$BUILDX_BUILDER" >/dev/null 2>&1 || \
    docker buildx create --name "$BUILDX_BUILDER" --driver docker-container --bootstrap
fi
for name in $(image_names); do
  df="$HERE/deploy/$(image_dockerfile "$name")"; tag="$(image_repo "$name"):latest"
  echo "$ build $tag  <- $df"
  if [ "$CROSS" = 1 ]; then
    # --provenance/--sbom=false: a plain single-arch image (AgentCore's containerUri rejects attestation lists)
    docker buildx build --builder "$BUILDX_BUILDER" --platform "$PLATFORM" --provenance=false --sbom=false \
      -f "$df" -t "$tag" --load "$HERE"
  else
    docker build --platform "$PLATFORM" -f "$df" -t "$tag" "$HERE"
  fi
  mb=$(( $(docker image inspect -f '{{.Size}}' "$tag") / 1024 / 1024 ))
  echo "built $tag ($PLATFORM, ${mb} MB)"
  if [ "$SUBSTRATE" = microvm ] && [ "$mb" -gt 2048 ]; then
    echo "!! $tag is over the 2048 MB microVM image cap — use a capacity-provider Instance (see demo_cpu_swebench, m5)"
  fi
done
log "done — next: bash 3_push_ecr.sh"
