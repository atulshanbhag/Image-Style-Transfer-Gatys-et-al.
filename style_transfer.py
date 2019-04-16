from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)

    if max_size is not None:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape is not None:
        image = image.resize(shape, Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image.to(device)


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = set(['0', '5', '10', '19', '28'])
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


def content_loss(x, y):
    return F.mse_loss(x, y)


def gram_matrix(x):
    return torch.einsum('bchw,bdhw->bcd', x, x)


def style_loss(x, y):
    gx = gram_matrix(x)
    gy = gram_matrix(y)
    return F.mse_loss(gx, gy)


def total_variance_loss(x):
    a = torch.abs((x[:, :, 1:, :] - x[:, :, :-1, :])).sum()
    b = torch.abs((x[:, :, :, 1:] - x[:, :, :, :-1])).sum()
    return torch.pow(a + b, 1.125)


def main(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    content = load_image(config.content, transform, max_size=config.max_size)
    style = load_image(config.style,
                       transform,
                       shape=[content.size(2),
                              content.size(3)])

    target = content.clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS([target], lr=config.lr)
    vgg = VGGNet().to(device).eval()

    content_features = vgg(content)
    style_features = vgg(style)

    for step in range(config.total_step):

        def closure():
            target_features = vgg(target)

            s_loss = 0
            c_loss = 0
            tv_loss = total_variance_loss(target) / (3 * content.size(2) *
                                                     content.size(3))
            for f1, f2, f3 in zip(target_features, content_features,
                                  style_features):
                _, c, h, w = f1.size()
                c_loss += content_loss(f1, f2)
                s_loss += style_loss(f1, f3) / (c * h * w)
                tv_loss += total_variance_loss(f1) / (c * h * w)

            loss = (config.content_weight * c_loss +
                    config.style_weight * s_loss +
                    config.total_variance_weight * tv_loss)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            print(
                '\rStep [{:04d}/{:04d}], Content Loss: {:7.4f}, Style Loss: {:10.4f}, Total Variance Loss: {:10.4f}'
                .format(step + 1, config.total_step, c_loss.item(),
                        s_loss.item(), tv_loss.item()),
                end='')

            return loss

        optimizer.step(closure)
        if (step + 1) % config.sample_step == 0:
            denorm = transforms.Normalize((-2.12, -2.04, -1.80),
                                          (4.37, 4.46, 4.44))
            img = target.cpu().detach().squeeze()
            img = denorm(img).clamp_(0, 1)
            torchvision.utils.save_image(
                img, 'results/output-{}.png'.format(step + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Image Style Transfer Using Convolutional Neural Networks, CVPR 2016, by Gatys et al.'
    )
    parser.add_argument('--content',
                        type=str,
                        required=True,
                        help='path to content image')
    parser.add_argument('--style',
                        type=str,
                        required=True,
                        help='path to style image')
    parser.add_argument(
        '--max_size',
        type=int,
        default=480,
        help='rescales image to maximum width or height, default=480')
    parser.add_argument(
        '--total_step',
        type=int,
        default=30,
        help='total no. of iterations of the algorithm, default=30')
    parser.add_argument(
        '--sample_step',
        type=int,
        default=5,
        help='save generated image after every sample step, default=5')
    parser.add_argument('--content_weight',
                        type=float,
                        default=1,
                        help='content loss hyperparameter, default=1')
    parser.add_argument('--style_weight',
                        type=float,
                        default=100,
                        help='style loss hyperparameter, default=100')
    parser.add_argument(
        '--total_variance_weight',
        type=float,
        default=0.01,
        help='total variance loss hyperparameter, default=0.01')
    parser.add_argument('--lr',
                        type=float,
                        default=1,
                        help='learning rate for L-BFGS, default=1')
    config = parser.parse_args()
    print(config)
    main(config)