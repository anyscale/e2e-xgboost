FROM anyscale/ray:2.44.1-slim-py312-cu123

# Set environment variable for uv
ENV RAY_RUNTIME_ENV_HOOK="ray._private.runtime_env.uv_runtime_env_hook.hook"

# Add your Debian packages here. Do not install unnecessary packages for a smaller image size to optimize build and cluster startup time.
# RUN sudo apt-get update -y \
#     && sudo apt-get install --no-install-recommends -y <package1> <package2> \
#     && sudo rm -f /etc/apt/sources.list.d/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
RUN uv sync

# Add other build commands here.
# RUN echo "Testing Ray import..." && python -c "import ray"
