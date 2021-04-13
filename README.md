# Saliency detection in human crowd images of different density levels using attention mechanism

The human visual system has the ability to rapidly identify and redirect attention to important visual information in high complexity scenes such as the human crowd. Saliency prediction in the human crowd scene is the process using computer vision techniques to imitate the human visual system, predicting which areas in a human crowd scene may attract human attention. However, it is a challenging task to identify which factors may attract human attention due to the high complexity of the human crowd scene. In this work, we propose Multiscale DenseNet — Dilated and Attention (MSDense-DAt), a convolutional neural network (CNN) using self-attention to integrate the result of knowledge-driven gaze in the human visual system to identify salient areas in the human crowd scene. Our method combines various state-of-the-art deep learning architectures to deal with the high complexity in human crowd image, such as multiscale DenseNet for multiscale deep features extraction, self-attention, and dilated convolution. Then the effectiveness of each component in our CNN architecture is evaluated by comparing different components combinations. Finally, the proposed method is further evaluated in different crowd density levels to appraise the effect of crowd density on model performance.

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
@article{nguyen2020saliency,
  title={Saliency detection in human crowd images of different density levels using attention mechanism},
  author={Nguyen, Minh Tri and Siritanawan, Prarinya and Kotani, Kazunori},
  journal={Signal Processing: Image Communication},
  volume={88},
  pages={115976},
  year={2020},
  publisher={Elsevier}
}
```
