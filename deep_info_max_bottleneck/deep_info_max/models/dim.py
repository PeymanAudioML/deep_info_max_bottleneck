
import numpy as np
import torch
import torch.nn as nn
from config.convnets import Convnet
import torch.nn.functional as F
import copy

class Permute(torch.nn.Module):
    """Module for permuting axes.

    """
    def __init__(self, *perm):
        """

        Args:
            *perm: Permute axes.
        """
        super().__init__()
        self.perm = perm

    def forward(self, input):
        """Permutes axes of tensor.

        Args:
            input: Input tensor.

        Returns:
            torch.Tensor: permuted tensor.

        """
        return input.permute(*self.perm)
    
class MI1x1ConvNet(nn.Module):
    """Simple custorm 1x1 convnet.

    """
    def __init__(self, n_input, n_units,):
        """

        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """

        super().__init__()

        self.block_nonlinear = nn.Sequential(
            nn.Conv2d(n_input, n_units, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_units),
            nn.ReLU(),
            nn.Conv2d(n_units, n_units, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.block_ln = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(n_units),
            Permute(0, 3, 1, 2)
        )

        self.linear_shortcut = nn.Conv2d(n_input, n_units, kernel_size=1,
                                         stride=1, padding=0, bias=False)

        # initialize shortcut to be like identity (if possible)
        if n_units >= n_input:
            eye_mask = np.zeros((n_units, n_input, 1, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
            self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def forward(self, x):
        """

            Args:
                x: Input tensor.

            Returns:
                torch.Tensor: network output.

        """

        h = self.block_ln(self.block_nonlinear(x) + self.linear_shortcut(x))
        return h


class NopNet(nn.Module):
    def __init__(self, norm_dim=None):
        super(NopNet, self).__init__()
        self.norm_dim = norm_dim
        return

    def forward(self, x):
        if self.norm_dim is not None:
            x_norms = torch.sum(x**2., dim=self.norm_dim, keepdim=True)
            x_norms = torch.sqrt(x_norms + 1e-6)
            x = x / x_norms
        return x
    

# Assuming ResNet, Convnet, NopNet, MI1x1ConvNet, compute_dim_loss, and sample_locations are defined elsewhere

class GlobalDIM(nn.Module):
    '''Global version of Deep InfoMax'''

    def __init__(self, encoder, config, task_idx=None, mi_units=2048):
        '''
        Args:
            encoder (nn.Module): The encoder network.
            config (dict): Configuration dictionary containing 'layers' and optionally 'local_task_idx'.
            task_idx (tuple): Indices where to do the local and global objectives.
            mi_units (int): Number of units for MI estimation.
        '''
        super(GlobalDIM, self).__init__()
        self.encoder = encoder
        self.mi_units = mi_units

        if task_idx is not None:
            self.task_idx = task_idx
        elif 'local_task_idx' not in config.keys():
            raise ValueError('No task_idx provided for GlobalDIM.')
        else:
            self.task_idx = config['local_task_idx']

        # Run a dummy forward pass to get the output sizes
        input_size = config.get('input_size', (3, 32, 32))  # Default input size
        X_dummy = torch.randn(1, *input_size)
        outs = self.encoder(X_dummy, return_all_activations=True)
        L, G = [outs[i] for i in self.task_idx]
        local_size = L.size()[1:]
        global_size = G.size()[1:]

        # For global DIM, we'll copy the layer hyperparameters for the encoder
        layers = copy.deepcopy(config['layers'])
        layers[-1] = dict(layer='linear', args=(mi_units,))

        if isinstance(encoder, Convnet):
            EncoderClass = Convnet
        else:
            raise NotImplementedError(f"Can't handle encoder of type {type(encoder)}")

        if len(local_size) == 1 or len(local_size) == 3:
            self.local_MINet = EncoderClass(local_size[::-1], layers=layers[self.task_idx[0]:])
        else:
            raise NotImplementedError("Local size not supported.")

        if len(global_size) == 1:
            if global_size[0] == mi_units:
                self.global_MINet = NopNet()
            else:
                self.global_MINet = EncoderClass(global_size, layers=layers[self.task_idx[1]:])
        elif len(global_size) == 3:
            if (global_size[1] == global_size[2] == 1) and global_size[0] == mi_units:
                self.global_MINet = NopNet()
            else:
                self.global_MINet = EncoderClass(global_size, layers=layers[self.task_idx[1]:])
        else:
            raise NotImplementedError("Global size not supported.")

    def forward(self, X):
        '''
        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            tuple: Encoded local and global features.
        '''
        outs = self.encoder(X, return_all_activations=True)
        L, G = [outs[i] for i in self.task_idx]

        L = self.local_MINet(L)
        G = self.global_MINet(G)

        N, local_units = L.size()[:2]
        L = L.view(N, local_units, -1)
        G = G.view(N, local_units, -1)
        return L, G

class LocalDIM(nn.Module):
    '''Local version of Deep InfoMax'''

    def __init__(self, encoder, config, task_idx=None, mi_units=2048, global_samples=None, local_samples=None):
        '''
        Args:
            encoder (nn.Module): The encoder network.
            config (dict): Configuration dictionary containing 'layers' and 'local_task_idx'.
            task_idx (tuple): Indices where to do the local and global objectives.
            mi_units (int): Number of units for MI estimation.
            global_samples (int): Number of samples over the global locations for each example.
            local_samples (int): Number of samples over the local locations for each example.
        '''
        super(LocalDIM, self).__init__()
        self.encoder = encoder
        self.mi_units = mi_units
        self.global_samples = global_samples
        self.local_samples = local_samples

        if task_idx is not None:
            self.task_idx = task_idx
        elif 'local_task_idx' not in config.keys():
            raise ValueError('No task_idx provided for LocalDIM.')
        else:
            self.task_idx = config['local_task_idx']

        # Run a dummy forward pass to get the output sizes
        input_size = config.get('input_size', (3, 32, 32))  # Default input size
        X_dummy = torch.randn(1, *input_size)
        outs = self.encoder(X_dummy, return_all_activations=True)
        L, G = [outs[i] for i in self.task_idx]
        local_size = L.size()[1:]
        global_size = G.size()[1:]

        if len(local_size) == 1 or len(local_size) == 3:
            self.local_MINet = MI1x1ConvNet(local_size[0], mi_units)
        else:
            raise NotImplementedError("Local size not supported.")

        if len(global_size) == 1:
            if global_size[0] == mi_units:
                self.global_MINet = NopNet()
            else:
                self.global_MINet = MI1x1ConvNet(global_size[0], mi_units)
        elif len(global_size) == 3:
            if (global_size[1] == global_size[2] == 1) and global_size[0] == mi_units:
                self.global_MINet = NopNet()
            else:
                self.global_MINet = MI1x1ConvNet(global_size[0], mi_units)
        else:
            raise NotImplementedError("Global size not supported.")

    def forward(self, X):
        '''
        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            tuple: Encoded local and global features.
        '''
        outs = self.encoder(X, return_all_activations=True)
        L, G = [outs[i] for i in self.task_idx]

        # Reshape global features as 1x1 feature maps if necessary
        global_size = G.size()[1:]
        if len(global_size) == 1:
            G = G[:, :, None, None]

        L = self.local_MINet(L)
        G = self.global_MINet(G)

        N, local_units = L.size()[:2]
        L = L.view(N, local_units, -1)
        G = G.view(N, local_units, -1)


        return L, G
    
if __name__ == '__main__':
    # Test the GlobalDIM and LocalDIM classes
    encoder = Convnet((3, 32, 32), layers=[dict(layer='conv', args=(64, 3, 1, 1)),
                                           dict(layer='conv', args=(128, 3, 1, 1)),
                                           dict(layer='conv', args=(256, 3, 1, 1)),
                                           dict(layer='conv', args=(512, 3, 1, 1),
                                                pool='avg', pool_args=(1,)),
                                           dict(layer='linear', args=(2048,))])

    # Test GlobalDIM
    global_dim = GlobalDIM(encoder, config={'layers': [dict(layer='linear', args=(2048,))],
                                           'local_task_idx': (3, 3)}, mi_units=2048)
    X = torch.randn(32, 3, 32, 32)
    L, G = global_dim(X)
    print(L.size(), G.size())

    # Test LocalDIM
    local_dim = LocalDIM(encoder, config={'layers': [dict(layer='linear', args=(2048,))],
                                         'local_task_idx': (3, 3)}, mi_units=2048)
    X = torch.randn(32, 3, 32, 32)
    L, G = local_dim(X)
    print(L.size(), G.size())


    
    