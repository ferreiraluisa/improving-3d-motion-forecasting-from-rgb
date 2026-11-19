#!/usr/bin/env bash
# Restore the "thesis" env from thesis_env.tar.gz and clone WHAM.
set -e

# Clone WHAM (with ViTPose)
git clone --recursive https://github.com/yohanshin/WHAM.git ~/WHAM

# Unpack the env
mkdir -p ~/.conda/envs/thesis
tar -xzf thesis_env.tar.gz -C ~/.conda/envs/thesis
~/.conda/envs/thesis/bin/conda-unpack

# Activate it
eval "$(conda shell.bash hook)"
conda activate thesis

# Reinstall ViTPose (not included in the tarball)
pip install -e ~/WHAM/third-party/ViTPose --no-deps

# Let torch load old checkpoints
conda env config vars set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1