# Runtime for the public API: its Python environment, locked by the repo's
# uv.lock. The image holds no code and no data; run.sh mounts a checkout
# read-only and the service runs its code, its target/release/simulator and
# its profiling/profile.db. The base is Ubuntu 24.04 because the simulator is
# built on such a host and links its libpython3.12. git reads the served commit.
FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 libpython3.12t64 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.19 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON=/usr/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
# run.sh passes only these two files as the build context. The service finds
# and runs the simulator binary through launcher code, hence the launcher group.
COPY pyproject.toml uv.lock /opt/servingstudio/
RUN cd /opt/servingstudio \
    && uv sync --frozen --no-install-project --no-default-groups --group public-api --group launcher \
    && rm -rf /root/.cache/uv
# The checkout is mounted read-only, so Python must not write bytecode into it.
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
