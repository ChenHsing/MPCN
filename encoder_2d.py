# -*- coding: utf-8 -*-

import torchvision.models
import torch



class Encoder2d(torch.nn.Module):
    def __init__(self, ):
        super(Encoder2d, self).__init__()

        # Layer Definition
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*[
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3,
            resnet.layer4
        ])[:6]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.resnet(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 256, 14, 14])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 256, 7, 7])

            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        image_features = image_features.view(-1,2048)
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 7, 7])
        return image_features


if __name__ == '__main__':
    encoder2d = Encoder2d()
    x = torch.rand(2,1,3,224,224)
    fe = encoder2d(x)
    print(fe.shape)

    # pos = torch.zeros(2048)
    # neg = torch.ones(2048)
    # pos = np.array(pos)
    # neg = np.array(neg)
    # print(fe[0].shape)
    # loss = encoder2d.optimize_params(fe[0],pos,neg)
    # print(loss)
