import os

# Enable uv on Ray
# https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#using-uv-for-package-management
os.environ["RAY_RUNTIME_ENV_HOOK"] = "ray._private.runtime_env.uv_runtime_env_hook.hook"
