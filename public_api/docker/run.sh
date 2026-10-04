#!/usr/bin/env bash
# Run the public API in Docker, restarted on failure and when the host reboots.
#
#   public_api/docker/run.sh BIND PORT     (re)build the image, (re)create the container
#
# The image carries only the Python environment (uv.lock's `public-api` and
# `launcher` groups). The container mounts this checkout read-only at the same
# path and serves its profiling/profile.db with its target/release/simulator,
# so build that first. To serve new code or data, update the checkout, rebuild
# the simulator and run this again.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BIND PORT" >&2
  exit 2
fi
bind=$1
port=$2
name=${PUBLIC_API_CONTAINER:-servingstudio-public-api}
image=${PUBLIC_API_IMAGE:-servingstudio-public-api}
repo=$(cd "$(dirname "$0")/../.." && pwd -P)

for path in "$repo/target/release/simulator" "$repo/profiling/profile.db"; do
  [[ -e $path ]] || { echo "$path is missing" >&2; exit 1; }
done

# The environment depends only on the lock, so the context is just these files.
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
cp "$repo/public_api/docker/Dockerfile" "$repo/pyproject.toml" "$repo/uv.lock" "$context/"
docker build -q -t "$image" "$context" >/dev/null

docker rm -f "$name" >/dev/null 2>&1 || true
# The service asks git for the commit it serves. A worktree's git directory
# lies outside the checkout, so it is mounted too.
mounts=(-v "$repo:$repo:ro")
git_dir=$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)
[[ $git_dir == "$repo"/* ]] || mounts+=(-v "$git_dir:$git_dir:ro")
docker run -d --name "$name" --restart unless-stopped \
  --user "$(id -u):$(id -g)" \
  "${mounts[@]}" \
  -w "$repo" -p "$bind:$port:$port" \
  "$image" \
  python -m public_api serve --bind 0.0.0.0 --port "$port" \
  --db "$repo/profiling/profile.db" --build-type release >/dev/null
echo "$name serving $repo on $bind:$port"
