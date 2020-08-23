from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):

    def __init__(self, embed_size, train_cnn=False):
        super(EncoderCNN, self).__init__()
        self.train_cnn = train_cnn
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, images):
        features = self.inception(images)
        for name, parameters in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                parameters.requires_grad = True
            else:
                parameters.requires_grad = False
        return self.dropout(self.relu(features))



