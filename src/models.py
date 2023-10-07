import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from stack import train as stack_train


class StackModel:

    def __init__(self, root, loss, lam, output_stack, detach=False):
        assert output_stack in [1, 2, 3, 4]
        self._output_stack = output_stack
        print(f"{loss}_{lam}_{detach}_False loading....")
        self._model = stack_train.Trainer.load_model(f"{root}/dependencies/stack", f"{loss}_{lam}_{detach}_False")
        self._model.eval()

    def __call__(self, x):
        # x: b x n x t x h x w
        x = x.cuda()
        out = self._model(x.unsqueeze(0), layer=self._output_stack)
        out = out[0].permute(1, 0, 2, 3).flatten(1, 3)

        return out.cpu()


class PrednetModel:

    INPUT_LEN = 10  # Seems like this is the cap enforced by the prednet model

    def __init__(self, root, output_stack):
        assert output_stack in ("E0", "E1", "E2", "E3")

        # Load model specific dependencies
        sys.path.append(os.path.join(root, "dependencies", "prednet"))
        from keras.models import model_from_json
        from prednet import PredNet

        self.output_stack = output_stack  # starts at 'E0'

        # Load the prednet model
        weights_dir = os.path.join(root, 'dependencies', 'prednet', 'model_data_keras2')
        weights_file = os.path.join(weights_dir, 'tensorflow_weights/prednet_kitti_weights.hdf5')
        json_file = os.path.join(weights_dir, 'prednet_kitti_model.json')

        # Load trained model
        f = open(json_file, 'r')
        json_string = f.read()
        f.close()

        prednet_config = model_from_json(json_string, custom_objects={'PredNet': PredNet})
        prednet_config.load_weights(weights_file)

        # Set the output to the provided layer
        layer_config = prednet_config.layers[1].get_config()
        layer_config['output_mode'] = output_stack
        self.prednet = PredNet(weights=prednet_config.layers[1].get_weights(), **layer_config)
        self.model = None

    def __call__(self, x):
        import keras
        from keras.layers import Input

        # x: chan x time x height x width

        if self.model is None:
            channels = 3

            h, w = x.shape[2], x.shape[3]
            if h == 73 and w == 73:  # Little fix to ensure we can base in data scaled at 0.66x
                h, w = 80, 80

            input_shape = [PrednetModel.INPUT_LEN, channels, h, w]
            print(input_shape)
            inputs = Input(shape=tuple(input_shape))
            predictions = self.prednet(inputs)
            self.model = keras.models.Model(inputs=inputs, outputs=predictions)

        h, w = x.shape[2], x.shape[3]
        if h == 73 and w == 73:  # Little fix to ensure we can base in data scaled at 0.66x
            x = F.pad(x, (3, 4, 3, 4))

        assert x.shape[2] % 2**3 == 0  # A requirement of the prednet model

        x = x.repeat(3, 1, 1, 1)  # Repeat the grayscale channel three times
        x = x.permute(1, 0, 2, 3)  # b x c x t x h x w -> b x t x c x h x w
        x = x.unsqueeze(0)  # Add batch dimension

        x = x[:, -PrednetModel.INPUT_LEN:, ]  # Only look at INPUT_LEN last frames
        x = self.model.predict(x.numpy(), PrednetModel.INPUT_LEN)  # Get hidden activity

        x = torch.from_numpy(x[0]).flatten(start_dim=1, end_dim=-1)  # time x neurons

        return x


class ImgModelBase:

    def __init__(self, n_warmup=None):
        self.n_warmup = n_warmup

    def __call__(self, x):
        # x: chan x time x height x width
        assert x.shape[0] == 1
        x = x[0]

        if self.n_warmup is not None:
            x = x[self.n_warmup:]

        activity_list = []
        for t in range(x.shape[0]):
            activity_list.append(self.model_output(x[t]))

        activity = torch.stack(activity_list)  # time x neurons

        # if self.n_warmup is not None:
        #     return F.pad(activity, (0, 0, self.n_warmup, 0))
        # else:
        return activity.detach().cpu()

    def model_output(self, x):
        raise NotImplementedError


class BWTModel(ImgModelBase):

    def __init__(self, root, n_warmup=None):
        super().__init__(n_warmup)
        # Load model specific dependencies
        from oct2py import octave
        octave.addpath(os.path.join(root, "dependencies", "bwt"))

        self.bwt = octave.bwt_v1_octave

    def model_output(self, x):
        h, w = x.shape
        target_dim = 3 ** int(np.ceil(np.log(h)/np.log(3)))
        h_pad = target_dim - h
        w_pad = target_dim - w

        xt = F.pad(x, (0, w_pad, 0, h_pad))
        return torch.from_numpy(self.bwt(xt.numpy())).flatten()


class VGGModel(ImgModelBase):

    def __init__(self, layer, n_warmup=None):
        super().__init__(n_warmup)
        self.vgg = models.vgg16(pretrained=True).cuda()
        self.layer = layer

    def model_output(self, x):
        x = x.cuda()
        if self.layer == '2.1':
            self.layer = 6
        elif self.layer == '2.2':
            self.layer = 8
        elif self.layer == '3.1':
            self.layer = 11
        elif self.layer == '3.2':
            self.layer = 13
        elif self.layer == '3.3':
            self.layer = 15

        x = x.unsqueeze(0).unsqueeze(0)  # Add back batch and channel dim
        x = x.repeat(1, 3, 1, 1)  # Repeat the grayscale image three times for the rgb channels

        with torch.no_grad():
            for i in range(self.layer + 1):
                x = self.vgg.features[i](x)

        return x.flatten()

