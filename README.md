


### Create conda env

```bash
# install mamba for faster installation (optional)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh

mamba create -p mani_env python=3.9
```

### Install dependencies

```bash
conda activate mani_env/
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
git clone https://github.com/haosulab/ManiSkill2.git

# Install stable baselines3 (if use sb-PPO)
pip install stable-baselines3[extra]

# install maniskill2
cd ManiSkill2 && pip install -e . && cd ..

```

### Install Maniskill learn (optional)

```bash
# Make sure the environment is activated
# First install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
# Anaconda Cloud
conda install pytorch3d -c pytorch3d-nightly
# pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# then install maniskill2-learn
git clone https://github.com/haosulab/ManiSkill2-Learn
cd ManiSkill2-Learn
pip install ninja
pip install -e .
pip install protobuf==3.19.0

ln -s ../ManiSkill2/data data # link the ManiSkill2 asset directory to ManiSkill2-Learn
# Alternatively, add `export MS2_ASSET_DIR={path_to_maniskill2}/data` to your bashrc file, so that the OS can find the asset directory no matter where you run MS2 envs.


```

```



## Trouble shooting
#### Gym installation error
Downgrade setuptools to 65.5.0
```bash
pip install setuptools==65.5.0 
```

if getting error:

wheel.vendored.packaging.requirements.InvalidRequirement: Expected end or semicolon (after version specifier)
opencv-python>=3.

```bash
pip install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40
```

if getting error:
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
kaggle 1.5.13 requires python-slugify, which is not installed.
```
pip3 install python-slugify
```