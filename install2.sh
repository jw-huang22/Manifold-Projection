#!/bin/bash
ENV_NAME="mpa-vit-env"
PYTHON_VERSION="3.9.18"
CONDA_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/anaconda"
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"


echo -e "\nCreating Conda Environment"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo -e "\nInstalling Conda Packages"
conda install -y \
  _libgcc_mutex=0.1=main \
  _openmp_mutex=5.1=1_gnu \
  ca-certificates=2024.7.2 \
  ld_impl_linux-64=2.38 \
  libffi=3.4.4 \
  libgcc-ng=11.2.0 \
  libgomp=11.2.0 \
  libstdcxx-ng=11.2.0 \
  openssl=3.0.14 \
  readline=8.2 \
  sqlite=3.41.2 \
  tk=8.6.12 \
  xz=5.4.6 \
  zlib=1.2.13 \
  pip=23.3.1

echo -e "\nInstalling Pip Packages "
export PIP_PROGRESS_BAR=on
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --retries 3 --timeout 120 -i $PYPI_MIRROR \
  accelerate==0.27.2 \
  aiohttp==3.9.3 \
  aiosignal==1.3.1 \
  antlr4-python3-runtime==4.9.3 \
  appdirs==1.4.4 \
  asttokens==2.4.1 \
  attrs==23.2.0 \
  certifi==2024.2.2 \
  charset-normalizer==3.3.2 \
  contourpy==1.3.0 \
  cycler==0.12.1 \
  datasets==2.18.0 \
  docker-pycreds==0.4.0 \
  docstring-parser==0.16 \
  evaluate==0.4.0 \
  executing==2.0.1 \
  fonttools==4.53.1 \
  frozenlist==1.4.1 \
  fsspec==2024.2.0 \
  gitdb==4.0.11 \
  gitpython==3.1.42 \
  gmpy2==2.2.1 \
  huggingface-hub==0.22.2 \
  hydra-core==1.3.2 \
  idna==3.6 \
  importlib-resources==6.4.4 \
  inquirerpy==0.3.4 \
  ipython==8.18.1 \
  jedi==0.19.1 \
  jinja2==3.1.3 \
  jsonargparse==4.21.1 \
  kiwisolver==1.4.7 \
  libnum==1.7.1 \
  lightning-utilities==0.11.7 \
  markdown-it-py==3.0.0 \
  markupsafe==2.1.5 \
  matplotlib==3.9.2 \
  mdurl==0.1.2 \
  mpmath==1.3.0 \
  multidict==6.0.5 \
  multiprocess==0.70.16 \
  naked==0.1.32 \
  networkx==3.2.1 \
  numpy==1.24.3 \
  omegaconf==2.3.0 \
  packaging==23.2 \
  pandas==2.0.1 \
  pathtools==0.1.2 \
  pexpect==4.9.0 \
  pfzy==0.3.4 \
  pillow==10.2.0 \
  protobuf==4.25.3 \
  psutil==5.9.8 \
  pyarrow==15.0.1 \
  pyarrow-hotfix==0.6 \
  pycryptodome==3.22.0 \
  pygments==2.17.2 \
  pyparsing==3.1.4 \
  pytorch-lightning==2.0.2 \
  regex==2023.12.25 \
  requests==2.32.2 \
  rich==13.8.0 \
  scikit-learn==1.5.1 \
  scipy==1.10.0 \
  seaborn==0.13.2 \
  sentencepiece==0.2.0 \
  sentry-sdk==1.41.0 \
  setproctitle==1.3.3 \
  setuptools==69.0.0 \
  shellescape==3.8.1 \
  smmap==5.0.1 \
  stack-data==0.6.3 \
  sympy==1.12 \
  tensorboardx==2.6.2.2 \
  timm==0.9.16 \
  tokenizers==0.15.2 \
  tqdm==4.66.2 \
  traitlets==5.14.1 \
  transformers==4.38.2 \
  triton==2.1.0 \
  torchmetrics==0.11.4 \
  typeshed-client==2.7.0 \
  typing-extensions==4.10.0 \
  tzdata==2024.1 \
  urllib3==2.2.1 \
  wandb==0.15.2 \
  wcwidth==0.2.13 \
  wheel==0.41.2 \
  xxhash==3.4.1 \
  yarl==1.9.4
