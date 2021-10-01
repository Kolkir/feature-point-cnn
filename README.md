# superpoint
This is a PyTorch implementation of CNN for image feature extraction it is based on training ideas applied for SuperPoint network but architecture is based on ResNet approach. Such architecture choice allowed to make model to train more faster with lower memory consumption, also this model can be trained in automatic mixed-precision mode and with a gradient accumulation technique that allows to use simpler GPUs to achieve normal training results.

Some algorithm implementations were inspired by the following repositories:
* https://github.com/rpautrat/SuperPoint
* https://github.com/magicleap/SuperPointPretrainedNetwork/

The main ideas are based on the following paper:

["SuperPoint: Self-Supervised Interest Point Detection and Description." Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich. ArXiv 2018.](https://arxiv.org/abs/1712.07629)
