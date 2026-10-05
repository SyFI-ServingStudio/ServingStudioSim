#!/usr/bin/env bash
# Run the public API in Docker, restarted on failure and when the host reboots.
#
#   public_api/docker/run.sh BIND PORT     (re)build the image, (re)create the container
#
# The image carries only the Python environment (uv.lock's `public-api` and
# `launcher` groups). The container mounts this checkout read-only at the same
# path and serves its profiling/profile.db with its target/release/simulator
# and analyze, so build those first. Predictions are kept for Read more in
# logs/public_api/predictions, simulations run in logs/public_api/simulations and
# uploaded workloads are kept in logs/public_api/workloads and every request is
# recorded in logs/public_api/access: the four writable mounts. The access log
# lives in the checkout, not in the container, so it survives this script
# replacing the container; Docker's own log of the process is capped. To serve
# new code or data, update the checkout, rebuild the binaries and run this
# again.
#
# The service is reached through the CSE web host's proxy (the Intro site's
# public/.htaccess), so the rate limits must count the client its
# X-Forwarded-For names, not the proxy. PUBLIC_API_FORWARDED_ALLOW_IPS lists the
# proxies trusted to name it (uvicorn's --forwarded-allow-ips, comma-separated);
# it defaults to new-rumble.cs.washington.edu, the CSE web host that forwards.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BIND PORT" >&2
  exit 2
fi
bind=$1
port=$2
name=${PUBLIC_API_CONTAINER:-servingstudio-public-api}
image=${PUBLIC_API_IMAGE:-servingstudio-public-api}
forwarded_allow_ips=${PUBLIC_API_FORWARDED_ALLOW_IPS:-128.208.3.117}
repo=$(cd "$(dirname "$0")/../.." && pwd -P)
hf_home=${HF_HOME:-$HOME/.cache/huggingface}

runs=$repo/logs/public_api/predictions
sims=$repo/logs/public_api/simulations
uploads=$repo/logs/public_api/workloads
access=$repo/logs/public_api/access
mkdir -p "$runs" "$sims" "$uploads" "$access"

tracegen=$repo/alignment/load_generator/req-frontend/target/release/tracegen
for path in "$repo/target/release/simulator" "$repo/target/release/analyze" "$tracegen" "$repo/profiling/profile.db"; do
  [[ -e $path ]] || { echo "$path is missing" >&2; exit 1; }
done

# The environment depends only on the lock, so the context is just these files.
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
cp "$repo/public_api/docker/Dockerfile" "$repo/pyproject.toml" "$repo/uv.lock" "$context/"
docker build -q -t "$image" "$context" >/dev/null

docker rm -f "$name" >/dev/null 2>&1 || true
# The service asks git for the commit it serves. A worktree's git directory
# lies outside the checkout, so it is mounted too. The presets' routing
# captures are read from the local hub cache only, so the cache is mounted too.
# The Analyzer listens on the container's own loopback, beside the service.
mounts=(-v "$repo:$repo:ro" -v "$runs:$runs" -v "$sims:$sims" -v "$uploads:$uploads" -v "$access:$access")
git_dir=$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)
[[ $git_dir == "$repo"/* ]] || mounts+=(-v "$git_dir:$git_dir:ro")
[[ -d $hf_home/hub ]] && mounts+=(-v "$hf_home/hub:$hf_home/hub:ro")
docker run -d --name "$name" --restart unless-stopped \
  --log-opt max-size=50m --log-opt max-file=4 \
  --user "$(id -u):$(id -g)" \
  "${mounts[@]}" -e HF_HOME="$hf_home" -e HF_HUB_OFFLINE=1 \
  -w "$repo" -p "$bind:$port:$port" \
  "$image" \
  python -m public_api serve --bind 0.0.0.0 --port "$port" \
  --db "$repo/profiling/profile.db" --build-type release \
  --runs-dir "$runs" --sims-dir "$sims" --workloads-dir "$uploads" --access-log-dir "$access" \
  --analyzer-port "$((port + 1))" \
  --forwarded-allow-ips "$forwarded_allow_ips" >/dev/null
echo "$name serving $repo on $bind:$port"
