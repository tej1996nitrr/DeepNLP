from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):

    def __init__(self, embed_size, train_cnn=False):
        super(EncoderCNN, self).__init__()
        self.train_cnn = train_cnn
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
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


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)  # vocab size represents each word in voabulary
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, features, captions):
        # features from CNN, captions from target captions(if teacher_forcing =1)
        embedding = self.dropout(self.embed(captions))
        embedding = torch.cat((features.unsqueeze(0), embedding), dim=0)
        hiddens, _ = self.lstm(embedding)  # embedding= captions+ features from image
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None  # hidden , cell states for lstms

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.unsqueeze(0))
                prediction = output.argmax(1)
                result_caption.append(prediction.item())
                x = self.decoderRNN.embed(prediction).unsqueeze(0)
                if vocabulary.itos[prediction.item()] == '<EOS>':
                    break
        return [vocabulary.itos[idx] for idx in result_caption]
