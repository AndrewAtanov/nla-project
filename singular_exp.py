from skimage.io import imread
from skimage.transform import resize
import os
import torch
import torchvision
import method
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vgg16')
parser.add_argument('--img_batch_path', default='img_batch.npy')
parser.add_argument('--gpu_id', default='1')
parser.add_argument('--pow_iter', default=10, type=int)
parser.add_argument('--fname', default='res.npy')
parser.add_argument('--nlayers', default=-1, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
d = torch.device('cuda')

layervssingular = []

if args.model == 'vgg16':
    net = torchvision.models.vgg16(pretrained=True)
    net.to(d)
    model = method.VGGWrapper(net, d)
elif args.model == 'vgg19':
    net = torchvision.models.vgg19(pretrained=True)
    net.to(d)
    model = method.VGGWrapper(net, d)
elif args.model == 'resnet34':
    net = torchvision.models.resnet34(pretrained=True)
    net.to(d)
    model = method.ResNetWrapper(net, d)

img_batch = np.load(args.img_batch_path)
tmp_img = torch.randn(1, 3, 224, 224).to(d)

if args.nlayers == -1:
    N = len(model.layers)
else:
    N = min(len(model.layers), args.nlayers)

for i in range(1, N+1):
    hidden_size = model.feature(tmp_img, i).shape[0]
    Adot, ATdot = method.get_batched_matvec(img_batch, model, i, hidden_size)
    x0 = (np.random.rand(3 * 224**2) - 0.5) / 255.
    x, s = method.power_method(x0, Adot, ATdot,  np.inf, 10, max_iter=args.pow_iter)
    layervssingular.append([i, s])
    print('layer {} value {}'.format(i, s))

np.save(args.fname, np.array(layervssingular))
