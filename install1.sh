#!/bin/bash
ENV_NAME="mpa-env"
PYTHON_VERSION="3.9.13"
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
  ca-certificates=2024.12.31 \
  ld_impl_linux-64=2.40 \
  libffi=3.3 \
  libgcc-ng=11.2.0 \
  libgomp=11.2.0 \
  libstdcxx-ng=11.2.0 \
  ncurses=6.4 \
  openssl=1.1.1 \
  readline=8.2 \
  setuptools=75.1.0 \
  sqlite=3.45.3 \
  tk=8.6.14 \
  wheel=0.44.0 \
  pip=24.2 \
  xz=5.4.6 \
  zlib=1.2.13 \

echo -e "\nInstalling Pip Packages "
export PIP_PROGRESS_BAR=on
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --retries 3 --timeout 120 -i $PYPI_MIRROR \
   accelerate==0.34.2 \
   aiohappyeyeballs==2.4.4 \
   aiohttp==3.11.11 \
   aiosignal==1.3.2 \
   async-timeout==5.0.1 \
   attrs==24.3.0 \
   certifi==2024.12.14 \
   charset-normalizer==3.4.1 \
   contourpy==1.3.0 \
   cycler==0.12.1 \
   datasets==3.0.0 \
   dill==0.3.8 \
   evaluate==0.4.3 \
   filelock==3.16.1 \
   fonttools==4.55.3 \
   frozenlist==1.5.0 \
   fsspec==2024.6.1 \
   huggingface-hub==0.27.1 \
   idna==3.10 \
   importlib-resources==6.5.2 \
   joblib==1.4.2 \
   kiwisolver==1.4.7 \
   matplotlib==3.9.4 \
   multidict==6.1.0 \
   multiprocess==0.70.16 \
   numpy==1.23.5 \
   packaging==24.2 \
   pandas==2.2.3 \
   pillow==11.1.0 \
   propcache==0.2.1 \
   psutil==6.1.1 \
   pyarrow==19.0.0 \
   pynvml==11.5.3 \
   pyparsing==3.2.1 \
   python-dateutil==2.9.0.post0 \
   pytz==2024.2 \
   pyyaml==6.0.2 \
   regex==2024.11.6 \
   requests==2.32.3 \
   safetensors==0.5.2 \
   scikit-learn==1.5.2 \
   scipy==1.13.1 \
   six==1.17.0 \
   threadpoolctl==3.5.0 \
   tokenizers==0.19.1 \
   tqdm==4.67.1 \
   transformers==4.44.2 \
   typing-extensions==4.12.2 \
   tzdata==2024.2 \
   urllib3==2.3.0 \
   xxhash==3.5.0 \
   yarl==1.18.3 \
   zipp==3.21.0 
pip install peft==0.14.0 --no-dependencies
