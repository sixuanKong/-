# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
import os
from utils.config import config
import torch.nn.functional as F



class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'LSTM_Resnet'
        self.train_path = 'data/' + dataset + '/train.txt'  # 训练集
        self.dev_path = 'data/' + dataset + '/dev.txt'  # 验证集
        self.test_path = 'data/' + dataset + '/test.txt'  # 测试集集

        self.class_list = [x.strip() for x in open(
            'data/' + dataset + '/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = config["vocab_path"]  # 词表
        if not os.path.exists('saved_dict/'): os.mkdir('saved_dict/')
        self.save_path = './saved_dict/' + dataset + '-' + self.model_name + '.ckpt'  # 模型训练结果
        self.embedding_pretrained = torch.tensor(
            np.load('./pretrained/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda:' + config["gpu"] if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = config["dropout"]  # 随机失活
        self.patience = config["patience"]
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = config["epochs"]  # epoch数
        self.batch_size = config["batch_size"]  # mini-batch大小
        self.pad_size = config["pad_size"]  # 每句话处理成的长度(短填长切)
        self.learning_rate = config["learning_rate"]  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数

        self.save_result = config["save_result"]


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# resnet50+lstm
class Model(nn.Module):
    def __init__(self, config, blocks):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc1 = nn.Linear(config.hidden_size * 2, 1)

        self.expansion = 4
        self.conv1 = Conv1(in_planes=3, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc2 = nn.Linear(2048, config.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.fc2 = nn.Linear(2048 + 1, config.num_classes)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, text, image):
        text, _ = text
        text = self.embedding(text)  # [batch_size, seq_len, embeding]=[128, 64, 300]
        text, _ = self.lstm(text)

        text = self.fc1(text[:, -1, :])  # 句子最后时刻的 hidden state

        image = self.conv1(image)

        image = self.layer1(image)
        image = self.layer2(image)
        image = self.layer3(image)
        image = self.layer4(image)

        image = self.avgpool(image)
        image = image.view(image.size(0), -1)
        # image = self.fc2(image)

        # x = x.reshape(1, -1)

        out = self.fc2(torch.cat((image, text), 1))

        return out
