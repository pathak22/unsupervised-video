"""
Transform images for compatibility with models trained with
https://github.com/facebook/fb.resnet.torch.

Usage in model prototxt:
layer {
  name: 'data_xform'
  type: 'Python'
  bottom: 'data_caffe'
  top: 'data'
  python_param {
    module: 'image_transform_layer'
    layer: 'TorchImageTransformLayer'
  }
}
"""

import caffe
import numpy as np


class TorchImageTransformLayer(caffe.Layer):
    def setup(self, bottom, top):
        # (1, 3, 1, 1) shaped arrays
        self.PIXEL_MEANS = \
            np.array([[[[0.485]],
                       [[0.456]],
                       [[0.406]]]])
        self.PIXEL_STDS = \
            np.array([[[[0.229]],
                       [[0.224]],
                       [[0.225]]]])
        top[0].reshape(*(bottom[0].shape))

    def forward(self, bottom, top):
        ims = bottom[0].data
        # 1. Permute BGR to RGB and normalize to [0, 1]
        ims = ims[:, [2, 1, 0], :, :] / 255.0
        # 2. Remove channel means
        ims -= self.PIXEL_MEANS
        # 3. Standardize channels
        ims /= self.PIXEL_STDS
        top[0].reshape(*(ims.shape))
        top[0].data[...] = ims

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
