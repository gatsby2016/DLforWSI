import cv2
import numpy as np
import torch

class CamExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        conv_output = None
        for name, module in self.model.features._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                conv_output = x
        return conv_output, x

    def forward_pass(self, x):
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()
        self.model = model.cuda()
        self.extractor = CamExtractor(self.model, target_layer_names)

    def __call__(self, input):
        features, output = self.extractor.forward_pass(input.cuda())
        index = np.argmax(output.cpu().data.numpy())

        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][index] = 1
        one_hot_output = one_hot_output.cuda()
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        output.backward(gradient=one_hot_output, retain_graph=True)

        grads_val = self.extractor.gradients.cpu().data.numpy()[0]

        target = features.cpu().data.numpy()[0]

        weights = np.mean(grads_val, axis=(1, 2))
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam