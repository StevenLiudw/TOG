


### Create conda env

```bash
mamba create -p mani_env python=3.9
```

### Install dependencies

```bash
conda activate mani_env
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
git clone https://github.com/haosulab/ManiSkill2.git

# Install stable baselines3
pip install stable-baselines3[extra]

# install maniskill2
cd ManiSkill2 && pip install -e .

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
