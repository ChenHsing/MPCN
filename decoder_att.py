# -*- coding: utf-8 -*-

import torch


class Decoder(torch.nn.Module):
    def __init__(self, ):
        super(Decoder, self).__init__()
        alpha = 0
        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(alpha)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(alpha)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(alpha)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.LeakyReLU(alpha)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, combine_features):
        # print(combine_features)
        gen_volume = combine_features.view(combine_features.size(0),512,2,2,2)
        gen_volume = self.layer1(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
        gen_volume = self.layer2(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        gen_volume = self.layer3(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
        gen_volume = self.layer4(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
        gen_volume = self.layer5(gen_volume)

        return gen_volume


if __name__ == '__main__':
    # x1 = torch.rand(2,1,128,4,4)
    x1 = torch.zeros(1,4096)
    decoder = Decoder()
    fe = decoder(x1)
    print(fe.shape)