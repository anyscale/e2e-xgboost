FROM anyscale/ray:2.44.1-slim-py312-cu123

# Add your Debian packages here. Do not install unnecessary packages for a smaller image size to optimize build and cluster startup time.
# RUN sudo apt-get update -y \
#     && sudo apt-get install --no-install-recommends -y <package1> <package2> \
#     && sudo rm -f /etc/apt/sources.list.d/*

# Copy the project code into the container
COPY --chown=ray:ray e2e_xgboost /home/ray/default
# Install dependencies
# Disable cache for a smaller image to optimize build and cluster startup time.
RUN python3 -m pip install --no-cache-dir requirements.txt

# Add other build commands here.
# RUN echo "Testing Ray import..." && python -c "import ray"
