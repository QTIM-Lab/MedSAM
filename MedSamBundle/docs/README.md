# MedSam Bundle

## Overview

## inference.yaml



## logging.conf

Just a standard logging file. Don't know much about it besides copy and pasted from MONAI Bundle tutorials

## metadata.json

The metadata file describing the bundle. Just standard information, think it is clear/you already know.

> Need to update the "network_data_format" and "outputs"

## train.yaml

run: is the thing that is being invoked. This is because:

In the source code for BundleTrainTask here: [https://docs.monai.io/projects/label/en/stable/_modules/monailabel/tasks/train/bundle.html#BundleTrainTask](https://docs.monai.io/projects/label/en/stable/_modules/monailabel/tasks/train/bundle.html#BundleTrainTask)

There is a line:

run_id = request.get("run_id", "run")

And I don't really know how to set run_id so just kept as run. -SK-

So you can add to the request body -BB-
```bash
{"val_split": 0.3}
or
{"run_id": "run_ex_id_0001"}
```
Then in `def __call__` you can access with `run_id = request.get("run_id", "run")` as Scott mentioned above.

Also in the docs we see BundleConfig at the top. Pretty much everything with key_ needs to be specified in this .yaml in the pattern specified by the BundleConfig. For example, that is why there is a train#trainer#max_epochs in one of the key_ functions, and then in the yamls is:

> BB - I think Scott meant BundleConstants if someone finds this. There are methods that start with "key_"

train:
    trainer:
        max_epochs: 3

The transforms are the proper training transforms that we want, which are coincidentally the ones I want for training in this simple environment because I don't want to add augmentations essentially; I just want to train on the original data and labels so we can see if it works, not be optimally generalizible to new data, which is usually the case. So, don't be concerned that transforms links to test_transforms; it's just a relic of naming conventions and the fact that in reality it would be train_transforms passed at

transforms: '$scripts.transforms.test_transform'

## scripts/dataloaders.py

## scripts/net.py

## scripts/train.py

## scripts/transforms.py

The original PyTorch transforms that my model needs to take an image and properly get a prediction under the same exact pre-processing it was trained on (normalization essentially.

I think MONAI normalization was acting SLIGHTLY different in normalization transform so would get BARELY different preds. But, was adding up and cause artificially high loss during training on a label generated from the prediction (i.e. 0.07 dice loss vs 0.02 (normal)). So needed to apply same transforms during training as in inference, and with same code to do it (PyTorch's, not MONAI's. Or, only use MONAI's, etc)

> BB - haven't touched yet...

# (Old) Your Model Name

Describe your model here and how to run it, for example using `metadata.json` and `train.yaml`:

```
# convenient to define the bundle's root in a variable
BUNDLE="./MedSamBundle"

# Tests
CONFIG_DEF_ENTRY_KEY=train_config
CONFIG_DEF_ENTRY_KEY=test_dataset
CONFIG_DEF_ENTRY_KEY=test_dataloader

# Infer
CONFIG_DEF_ENTRY_KEY=infer
CONFIG_FILE=$BUNDLE/configs/inference.yaml

# Train
CONFIG_DEF_ENTRY_KEY=train
CONFIG_FILE=$BUNDLE/configs/train.yaml

DATASET_CSV=/sddata/data/MedSAM/public_test_data_01_10_2023/image_key_cf.csv

# Note: --dataset_csv is made up config def entry key accessible with @dataset_csv

python -m monai.bundle run $CONFIG_DEF_ENTRY_KEY \
  --meta_file $BUNDLE/configs/metadata.json \
  --config_file $CONFIG_FILE \
  --dataset_csv $DATASET_CSV \
  --bundle_root $BUNDLE
```
