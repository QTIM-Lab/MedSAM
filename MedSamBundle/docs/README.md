# MedSam Bundle

## Overview

## inference.yaml

pre_transforms aren't even being used, just a placeholder. Being overwritten manually in SegmentationBundleInferTask's call by hard-coding the PyTorch transforms I wanted. This needs further investigation

post_transforms take the model prediction, which are logits, and then applies the sigmoid activation to turn them into probabilities, then argmax's over the channel with the highest probability to make the prediction as that channel (i.e. "disc" channel for pixel (245,253)). The result is still a (h,w,n_classes) shaped output, but only one value of n_classes at any (h,w) pixel is 1, rest is 0. So you get a very rigid RGB image

Image and pred are just keys needed for the monai transforms, as those transforms operate on a dictionary, and the keys are the things it is operating on. Which is why in the SegmentationBundleInfer task, you see me need to directly access the data dict's keys to pass those values to PyTorch transforms, since PyTorch doesn't have that functionality directly

MONAI and PyTorch transforms are similar but can vary slightly, but for sigmoid it's a more straightforward operation compared to their normalization or scaling. So for post tranforms, it was ok.

## logging.conf

Just a standard logging file. Don't know much about it besides copy and pasted from MONAI Bundle tutorials

## metadata.json

The metadata file describing the bundle. Just standard information, think it is clear/you already know.

## train.yaml

run: is the thing that is being invoked. This is because:

In the source code for BundleTrainTask here: [https://docs.monai.io/projects/label/en/stable/_modules/monailabel/tasks/train/bundle.html#BundleTrainTask](https://docs.monai.io/projects/label/en/stable/_modules/monailabel/tasks/train/bundle.html#BundleTrainTask)

There is a line:

run_id = request.get("run_id", "run")

And I don't really know how to set run_id so just kept as run.

Also in the docs we see BundleConfig at the top. Pretty much everything with key_ needs to be specified in this .yaml in the pattern specified by the BundleConfig. For example, that is why there is a train#trainer#max_epochs in one of the key_ functions, and then in the yamls is:

train:
    trainer:
        max_epochs: 3

The transforms are the proper training transforms that we want, which are coincidentally the ones I want for training in this simple environment because I don't want to add augmentations essentially; I just want to train on the original data and labels so we can see it works, not be optimally generalizible to new data, which is usually the case. So, don't be concerned that transforms links to test_transforms; it's just a relic of naming conventions and the fact that in reality it would be train_transforms passed at

transforms: '$scripts.transforms.test_transform'

## scripts/dataloaders.py

Create dataloader with transforms that worked for my original code

## scripts/net.py

The network definition for the Huggingface's implementaion of Segformer model, with the extra other functions to allow my to load the state dictionary properly (load the model weights essentially) even though I don't do a generic PyTorch nn.module but instead use this Huggingface model, have to do this if I want a nn.Module class wrapper around it to work normally

## scripts/train.py

The original training code for my model, trimmed down to not include outputs during training (because demo, and because we do literally 1 epoch for now, and would need to figure out where to save the outputs etc)

## scripts/transforms.py

The original PyTorch transforms that my model needs to take an image and properly get a prediction under the same exact pre-processing it was trained on (normalization essentially.

I think MONAI normalization was acting SLIGHTLY different in normalization transform so would get BARELY different preds. But, was adding up and cause artificially high loss during training on a label generated from the prediction (i.e. 0.07 dice loss vs 0.02 (normal)). So needed to apply same transforms during training as in inference, and with same code to do it (PyTorch's, not MONAI's. Or, only use MONAI's, etc)

# (Old) Your Model Name

Describe your model here and how to run it, for example using `inference.json`:

```
python -m monai.bundle run \
  --meta_file /path/to/bundle/configs/metadata.json \
  --config_file /path/to/bundle/configs/inference.json \            --dataset_dir ./input \
  --bundle_root /path/to/bundle
```
