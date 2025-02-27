mkdir cache
mkdir cache/avici
mkdir cache/avici/linear
mkdir cache/avici/linear/checkpoint
mkdir cache/avici/rff
mkdir cache/avici/rff/checkpoint

wget https://huggingface.co/larslorch/avici/resolve/main/neurips-linear/kwargs.json -P cache/avici/linear/checkpoint/
wget https://huggingface.co/larslorch/avici/resolve/main/neurips-linear/checkpoint_0300000.pkl -P cache/avici/linear/checkpoint/
# AVICI requires this file to be present twice
cp cache/avici/linear/checkpoint/kwargs.json cache/avici/linear/kwargs.json

wget https://huggingface.co/larslorch/avici/resolve/main/neurips-rff/kwargs.json -P cache/avici/rff/checkpoint/
wget https://huggingface.co/larslorch/avici/resolve/main/neurips-rff/checkpoint_0300000.pkl -P cache/avici/rff/checkpoint/
# AVICI requires this file to be present twice
cp cache/avici/rff/checkpoint/kwargs.json cache/avici/rff/kwargs.json