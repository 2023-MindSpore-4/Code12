# /bin/bash

# mindformer
git clone -b dev https://gitee.com/mindspore/mindformers.git
git checkout af3e4555789cd5eb89578e9d18d9afd80f02fac0
cd mindformers
bash build.sh
cd -

pip install -r requirements.txt

# data
cd pretrain
pip install en_core_web_sm-3.6.0.tar.gz
cd -

python pre_install.py