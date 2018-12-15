import numpy as np
import torch
import torchvision


def tonp(x):
    return x.cpu().detach().numpy()


class BaseWrapper(object):
    """ Wrapper for NN"""
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def feature(self, x, i):
        assert i <= len(self.layers)
        f = x
        for self.layer in self.layers[:i]:
            f = self.layer(f)
        f = f.flatten()

        return f

    def jacobi_matvec(self, x, i):
        """ return matvec for J_i(x) and transposed one """
        f = self.feature(x, i)
        g = torch.autograd.grad([f @ v2], [x], retain_graph=True, create_graph=True)[0]
        JT_v2 = g.flatten()

        v2.data = torch.zeros_like(v2.data)
        J_v1 = torch.autograd.grad([JT_v2 @ v1], [v2])[0]

        return J_v1, JT_v2

    def get_matvec_AT(self, img, i):
        x = torch.FloatTensor(img.flatten()).to(self.device)
        x.requires_grad = True
        x_img = x.reshape((1, 3, 224, 224))

        def mv(v2):
            v2 = torch.FloatTensor(v2).to(self.device)
            f = self.feature(x_img, i)
            JT_v2 = torch.autograd.grad([f @ v2], [x])[0]
            return tonp(JT_v2)

        return mv

    def get_matvec_A(self, img, i):
        x = torch.FloatTensor(img.flatten()).to(self.device)
        x.requires_grad = True
        x_img = x.reshape((1, 3, 224, 224))

        def mv(v1):
            f = self.feature(x_img, i)
            v2 = torch.zeros_like(f, requires_grad=True).to(self.device)
            v1 = torch.FloatTensor(v1).to(self.device)
            JT_v2 = torch.autograd.grad([f @ v2], [x], retain_graph=True, create_graph=True)[0]
            J_v1 = torch.autograd.grad([JT_v2 @ v1], [v2])[0]
            return tonp(J_v1)

        return mv


class VGGWrapper(BaseWrapper):
    def __init__(self, net, device):
        super(VGGWrapper, self).__init__(net, device)
        self.layers = self.net.features


class ResNetWrapper(BaseWrapper):
    def __init__(self, net, device):
        super(VGGWrapper, self).__init__(net, device)
        self.layers = self.list_childrens(self.net)[:-1]

    def list_childrens(self, module):
        res = []
        if isinstance(module, torch.nn.modules.container.Sequential):
            for c in module.children():
                res = res + self.list_childrens(c)
        elif isinstance(module, self.net.__class__):
            for c in module.children():
                res = res + self.list_childrens(c)
        else:
            res.append(module)
        return res


def psi(x, r):
    return np.sign(x) * np.power(np.abs(x), r-1)


def power_method(x0, matvec_A, matvec_AT, p, q, max_iter=1000):
    """ return (p, q) singular value of A """
    if p == np.inf:
        x = np.sign(x0)
    else:
        x = x0 / np.linalg.norm(x0, ord=p)
        p_hat = 1.0 / (1.0 - 1.0 / p)

    for _ in range(max_iter):
        Ax = matvec_A(x)
        if p == np.inf:
            x = np.sign(matvec_AT(psi(Ax, q)))
        else:
            Sx = psi(matvec_AT(psi(Ax, q)), p_hat)
            x = Sx / np.linalg.norm(Sx, ord=p)

        s = np.linalg.norm(Ax, ord=q)

    return x, s


def get_batched_matvec(imgs_batch, model, layer, hidden_size):
    """ construct matvec and return pertrubation wrt imgs """
    def Amv(x):
        res = []
        for i, img in enumerate(imgs_batch):
            matvec_A = model.get_matvec_A(img, layer)
            res.append(matvec_A(x))

        return np.concatenate(res)

    def ATmv(x):
        res = 0.
        for i, img in enumerate(imgs_batch):
            idx = i * hidden_size
            matvec_AT = model.get_matvec_AT(img, layer)
            res += matvec_AT(x[idx:idx + hidden_size])

        return res

    return Amv, ATmv
