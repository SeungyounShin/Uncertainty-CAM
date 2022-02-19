# UCAM


## Datasets
It assumes datasets are located in data root which can be modified in the data section of a config file. One way to set it up is to use a symlink as follows,
```bash
mkdir data && cd data
ln -s path/to/data/dir
```


## Setup
To setup the pipeline including creating additional files for dataset and installing required packages, run the command below.
```bash
bash scripts/setup.sh --data_setup
```


## Configs
Configs are written in the form of yaml. Please refer to the configs/default.yml for the details about how to structure configs.


## Train, val and test
The command below automatically trains a model for the given epochs and validates the model every epoch. Once training is done, it runs test on the best checkpoint from validation.
```bash
bash scripts/run.sh --config_path configs/default.yml --save_dir path/to/save/dir
```

## GradCAM
To modify the layer where gradcam is extract, please change the line 37-42 in
```bash
src/core/cams/gradcam
```
In the future, it will be reflected in a config.

## Tensorboard
Images with localization results are saved in save_dir. To check the results, run the following command with your choice of port number.
```bash
tensorbaord --log_dir path/to/save/dir --port 7015 --bind_all
```

