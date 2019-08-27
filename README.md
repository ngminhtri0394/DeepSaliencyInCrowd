# Saliency detection in human crowd image in different density levels using attention mechanism

The process using computer vision techniques to imitate the human visual system, predicting which areas on an image may attract human attention, is known as "saliency prediction". Saliency prediction in human crowd has important applications in human-computer interaction, security, multimedia. In this work, we proposed Multiscale DenseNet – Dilated and Attention (MSDense-DAt), a convolutional network for saliency prediction in crowd RGB data. Our method combines various state-of-the-art deep learning architectures  to deal with a crowd scene with varying object sizes, and complex composition of objects in human crowd image, such as multiscale DenseNet for multiscale deep features extraction, self-attention to imitate short-term knowledge driven gaze of human visual system, and dilated convolution. The overall performance is evaluated in human crowd image. Then the approach is evaluated in different crowd density levels to appraise the effect of crowd density on model performance. The result reveals that our approach overperforms previous saliency in crowd models. The self-attention mechanism can further enhance saliency model in high density crowd level.

## Usage
Environment:
```
Python 3
Tensorflow GPU 1.10.0
```

Ten-fold cross validation: 
```
main.py train 
```

Test with pre-trained weight
```
main.py test \path\to\pretrain\weight.h5 \path\to\input\image
```

Change model 
```
Change the "net" parameter in utils/configs.py
```

## Pre-trained weight

Pre-trained weight by training the Eyecrowd dataset.

[MSDensenet-DAt](https://drive.google.com/open?id=1lLCpbs4ZS4OwsR4wlvEaYS2Z7qeOnTfI)

[MSDense-D](https://drive.google.com/open?id=1APq0gCAlGDjT6a4Yq71CelAa24mYafPL)

[TSDense-D](https://drive.google.com/open?id=1KvZp8ETJxSv6Lm37wLh7OWldtk3L7GNj)

[MSDense](https://drive.google.com/open?id=1sr5aji-4qbrQff2QSwMFaX_6FfBpQnHL)

[Dense-D](https://drive.google.com/open?id=1KOBSyWctw9Lhyay90sKbgy0rGvggDlsl)

[Dense](https://drive.google.com/open?id=1BATVlsl-eOWhPedGEghU9_BY9M0WcRkm)

## Citation
```
@article{tri2019incrowd,
  title={Saliency detection in human crowd images of different density levels using attention mechanism},
  author={Tri, Nguyen Minh and Prarinya, Siritanawan and Kotani, Kazunori},
  year={2019}
}
```
