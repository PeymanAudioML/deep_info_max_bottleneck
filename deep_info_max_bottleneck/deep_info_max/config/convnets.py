'''Convnet encoder module.

'''

import copy

import torch
import torch.nn as nn

def get_nonlinearity(nonlinearity=None):
    if not nonlinearity:
        pass
    elif callable(nonlinearity):
        if nonlinearity == nn.LeakyReLU:
            nonlinearity = nonlinearity(0.02, inplace=True)
    elif hasattr(nn, nonlinearity):
        nonlin_ = nonlinearity
        nonlinearity = getattr(nn, nonlinearity)
        if nonlin_ == 'LeakyReLU':
            nonlinearity = nonlinearity(0.02, inplace=True)
        elif nonlin_ == 'ReLU':
            nonlinearity = nonlinearity(inplace=True)
        else:
            nonlinearity = nonlinearity()
    elif hasattr(nn.functional, nonlinearity):
        nonlinearity = getattr(nn.functional, nonlinearity)
    else:
        raise ValueError(nonlinearity)
    return nonlinearity

class View(torch.nn.Module):
    """Basic reshape module.

    """
    def __init__(self, *shape):
        """

        Args:
            *shape: Input shape.
        """
        super().__init__()
        self.shape = shape

    def forward(self, input):
        """Reshapes tensor.

        Args:
            input: Input tensor.

        Returns:
            torch.Tensor: Flattened tensor.

        """
        return input.view(*self.shape)


class Unfold(torch.nn.Module):
    """Module for unfolding tensor.

    Performs strided crops on 2d (image) tensors. Stride is assumed to be half the crop size.

    """
    def __init__(self, img_size, fold_size, stride=None, final_size=None):
        """

        Args:
            img_size: Input size.
            fold_size: Crop size.
        """
        super().__init__()

        fold_stride = stride or fold_size // 2
        self.fold_size = fold_size
        self.fold_stride = fold_stride
        self.n_locs = final_size or (2 * int(img_size / fold_size) - 1)
        self.unfold = torch.nn.Unfold((self.fold_size, self.fold_size),
                                      stride=(self.fold_stride, self.fold_stride))

    def forward(self, x):
        """Unfolds tensor.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Unfolded tensor.

        """
        N = x.size(0)
        x = self.unfold(x).reshape(N, -1, self.fold_size, self.fold_size, self.n_locs * self.n_locs)\
            .permute(0, 4, 1, 2, 3)\
            .reshape(N * self.n_locs * self.n_locs, -1, self.fold_size, self.fold_size)
        return x.contiguous()


class Fold(torch.nn.Module):
    """Module (re)folding tensor.

    Undoes the strided crops above. Works only on 1x1.

    """
    def __init__(self, n_locs):
        """

        Args:
            n_locs: Number of locations in output.
        """
        super().__init__()
        self.n_locs = n_locs

    def forward(self, x):
        """(Re)folds tensor.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Refolded tensor.

        """
        x_shape = x.size()[1:]

        if len(x_shape) == 3:
            dim_c, dim_x, dim_y = x_shape
        else:
            dim_c = x_shape[0]
            dim_x = 1
            dim_y = 1
        x = x.reshape(-1, self.n_locs * self.n_locs, dim_c, dim_x * dim_y)
        x = x.reshape(-1, self.n_locs * self.n_locs, dim_c, dim_x * dim_y)\
            .permute(0, 2, 3, 1)\
            .reshape(-1, dim_c * dim_x * dim_y, self.n_locs, self.n_locs).contiguous()
        return x


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


class Expand2d(torch.nn.Module):
    """Expands 2d tensor to 4d

    """

    def forward(self, input):
        """Expands tensor to 4d.

        Args:
            input: Input tensor.

        Returns:
            torch.Tensor: expanded tensor.

        """
        size = input.size()

        if len(size) == 4:
            return input
        elif len(size) == 2:
            return input[:, None, None, :]
        else:
            raise ValueError()

class Convnet(nn.Module):
    '''Basic convnet convenience class.

    Attributes:
        layers: nn.Sequential of layers with batch norm,
            dropout, nonlinearity, etc.
        shapes: list of output shapes for every layer..

    '''

    _supported_types = ('linear', 'conv', 'tconv', 'reshape', 'flatten', None)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.create_layers(*args, **kwargs)

    def create_layers(self, shape, layers=None):
        '''Creates layers

        Args:
            shape: Shape of input.
            layers: list of layer arguments.
        '''

        self.layers, self.shapes = self.create_sequential(shape, layers=layers)

    def create_sequential(self, shape, layers=None):
        '''Creates a sequence of layers.

        Args:
            shape: Input shape.
            layers: list of layer arguments.

        Returns:
            nn.Sequential: a sequence of convolutional layers.

        '''

        modules = nn.Sequential()
        layers = layers or []
        layers = copy.deepcopy(layers)
        shapes = []

        for i, layer in enumerate(layers):
            layer_type = layer.pop('layer', None)

            name = 'layer{}'.format(i)
            block = nn.Sequential()

            shape = self.handle_layer(block, shape, layer, layer_type)
            shape = self.finish_block(block, shape, **layer)
            if len(block) == 1:
                block = block[0]
            shapes.append(shape)

            modules.add_module(name, block)

        return modules, shapes

    def handle_layer(self, block, shape, layer, layer_type):
        '''Handles the layer arguments and adds layer to the block.

        Args:
            block: nn.Sequential to add modules to.
            shape: Shape of the input.
            layer: Layer arguments.
            layer_type: Type of layer.

        Returns:
            tuple: Output shape.

        '''
        args = layer.pop('args', None)
        if layer_type == 'linear':
            if len(shape) == 3:
                dim_x, dim_y, dim_out = shape
                shape = (dim_x * dim_y * dim_out,)
                block.add_module('flatten', View(-1, shape[0]))
            bn = layer.get('bn', False)
            bias = layer.pop('bias', None)
            init = layer.pop('init', None)
            init_args = layer.pop('init_args', {})
            shape = self.add_linear_layer(block, shape, args=args, bn=bn, bias=bias, init=init, init_args=init_args)
        elif layer_type == 'conv':
            if len(shape) == 1:
                shape = (1, 1, shape[0])
                block.add_module('expand', Expand2d())
            bn = layer.get('bn', False)
            bias = layer.pop('bias', None)
            init = layer.pop('init', None)
            init_args = layer.pop('init_args', {})
            shape = self.add_conv_layer(block, shape, args=args, bn=bn, bias=bias, init=init, init_args=init_args)
        elif layer_type == 'tconv':
            if len(shape) == 1:
                raise ValueError('Transpose conv needs 4d input')
            bn = layer.get('bn', False)
            bias = layer.pop('bias', True)
            shape = self.add_tconv_layer(block, shape, args=args, bn=bn, bias=bias)
        elif layer_type == 'flatten':
            if len(shape) == 3:
                dim_x, dim_y, dim_out = shape
                shape = (dim_x * dim_y * dim_out,)
            block.add_module(layer_type, View(-1, shape[0]))
        elif layer_type == 'reshape':
            if args is None:
                raise ValueError('reshape needs args')
            new_shape = args
            dim_new = 1
            dim_out = 1
            for s in new_shape:
                dim_new *= s
            for s in shape:
                dim_out *= s
            if dim_new != dim_out:
                raise ValueError('New shape {} not compatible with old shape {}.'
                                 .format(new_shape, shape))
            block.add_module(layer_type, View((-1,) + new_shape))
            shape = new_shape[::-1]
        elif layer_type is None:
            pass
        else:
            raise NotImplementedError(
                'Layer {} not supported. Use {}'.format(layer_type, self._supported_types))
        return shape

    def add_conv_layer(self, block, shape, args=None, bn=False, bias=None, init=None, init_args=None):
        '''Adds a convolutional layer to the block.

        Args:
            block: nn.Sequential to add conv layer to.
            shape: Shape of the input.
            args: conv layer arguments (n_units, filter size, stride, padding)
            bn (bool): Batch normalization.
            bias (bool): Controls bias in layer.
            init: Initialization of layer.
            init_args: Arguments for initialization.

        Returns:
            tuple: Output shape.

        '''
        dim_x, dim_y, dim_in = shape
        try:
            dim_out, f, s, p = args
        except:
            raise ValueError('args must be provided for conv layer and in format '
                             '`(depth, kernel size, stride, padding)`')

        if bias is None:
            bias = not (bn)
        conv = nn.Conv2d(dim_in, dim_out, kernel_size=f, stride=s, padding=p, bias=bias)
        if init:
            init = getattr(nn.init, init)
            init(conv.weight, **init_args)
        block.add_module('conv', conv)
        dim_x, dim_y = self.next_conv_size(dim_x, dim_y, f, s, p)

        return (dim_x, dim_y, dim_out)

    def add_tconv_layer(self, block, shape, args=None, bn=False, bias=None):
        '''Adds a transpose convolutional layer to the block.

        Args:
            block: nn.Sequential to add tconv layer to.
            shape: Shape of the input.
            args: tconv layer arguments (n_units, filter size, stride, padding)
            bn (bool): Batch normalization.
            bias (bool): Controls bias in layer.

        Returns:
            tuple: Output shape.

        '''

        dim_x, dim_y, dim_in = shape
        try:
            dim_out, f, s, p = args
        except:
            raise ValueError('args must be provided for tconv layer and in format '
                             '`(depth, kernel size, stride, padding)`')

        if bias is None:
            bias = not (bn)
        tconv = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=f, stride=s, padding=p, bias=bias)
        block.add_module('tconv', tconv)
        dim_x, dim_y = self.next_tconv_size(dim_x, dim_y, f, s, p)

        return (dim_x, dim_y, dim_out)

    def add_linear_layer(self, block, shape, args=None, bn=False, bias=None, init=None, init_args=None):
        '''Adds a linear layer

        Args:
            block: nn.Sequential to add linear layer to.
            shape: Shape of the input.
            args: linear layer arguments (n_units,)
            bn (bool): Batch normalization.
            bias (bool): Controls bias in layer.
            init: Initialization of layer.
            init_args: Arguments for initialization.

        Returns:
            tuple: Output shape.

        '''

        try:
            dim_out, = args
        except:
            raise ValueError('args must be provided for fully-connected layer and in format '
                             '`(depth,)`')

        dim_in, = shape
        if bias is None:
            bias = not (bn)
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        if init:
            init = getattr(nn.init, init)
            init(layer.weight, **init_args)
        block.add_module('fc', layer)

        return (dim_out,)

    def finish_block(self, block, shape, bn=False, ln=False, do=False, act=None, pool=None):
        '''Finishes a block.

        Adds batch norm, dropout, activation, pooling.

        Args:
            block (nn.Sequential): Block to add conv layer to.
            shape (tuple): Shape of the input.
            bn (bool): Batch normalization.
            ln (bool): Layer normalization.
            do (float): Dropout.
            act (str): Activation.
            pool (tuple): Pooling. In format (pool type, kernel size, stride).

        Returns:

        '''
        if len(shape) == 1:
            BN = nn.BatchNorm1d
            DO = nn.Dropout
        elif len(shape) == 3:
            BN = nn.BatchNorm2d
            DO = nn.Dropout2d
        else:
            raise NotImplementedError('Shape {} not supported'.format(shape))
        LN = nn.LayerNorm

        if ln and bn:
            raise ValueError('Use only one sort of normalization.')

        dim_out = shape[-1]

        if do:
            block.add_module('do', DO(p=do))
        if bn:
            block.add_module('bn', BN(dim_out))
        if ln:
            block.add_module('ln', LN(dim_out))

        if act:
            nonlinearity = get_nonlinearity(act)
            block.add_module(nonlinearity.__class__.__name__, nonlinearity)

        if pool:
            if len(shape) == 1:
                raise ValueError('Cannot pool on 1d tensor.')
            (pool_type, kernel, stride) = pool
            Pool = getattr(nn, pool_type)
            block.add_module('pool', Pool(kernel_size=kernel, stride=stride))
            dim_x, dim_y, dim_out = shape
            dim_x, dim_y = self.next_conv_size(dim_x, dim_y, kernel, stride, 0)
            shape = (dim_x, dim_y, dim_out)

        return shape

    def next_conv_size(self, dim_x, dim_y, k, s, p):
        '''Infers the next size of a convolutional layer.

        Args:
            dim_x: First dimension.
            dim_y: Second dimension.
            k: Kernel size.
            s: Stride.
            p: Padding.

        Returns:
            (int, int): (First output dimension, Second output dimension)

        '''

        def infer_conv_size(w, k, s, p):
            '''Infers the next size after convolution.

            Args:
                w: Input size.
                k: Kernel size.
                s: Stride.
                p: Padding.

            Returns:
                int: Output size.

            '''
            x = (w - k + 2 * p) // s + 1
            return x

        if isinstance(k, int):
            kx, ky = (k, k)
        else:
            kx, ky = k

        if isinstance(s, int):
            sx, sy = (s, s)
        else:
            sx, sy = s

        if isinstance(p, int):
            px, py = (p, p)
        else:
            px, py = p
        return (infer_conv_size(dim_x, kx, sx, px),
                infer_conv_size(dim_y, ky, sy, py))

    def next_tconv_size(self, dim_x, dim_y, k, s, p):
        '''Infers the next size of a transpose convolutional layer.

        Args:
            dim_x: First dimension.
            dim_y: Second dimension.
            k: Kernel size.
            s: Stride.
            p: Padding.

        Returns:
            (int, int): (First output dimension, Second output dimension)

        '''

        def infer_conv_size(w, k, s, p):
            '''Infers the next size after convolution.

            Args:
                w: Input size.
                k: Kernel size.
                s: Stride.
                p: Padding.

            Returns:
                int: Output size.

            '''
            x = s * (w - 1) - 2 * p + k
            return x

        if isinstance(k, int):
            kx, ky = (k, k)
        else:
            kx, ky = k

        if isinstance(s, int):
            sx, sy = (s, s)
        else:
            sx, sy = s

        if isinstance(p, int):
            px, py = (p, p)
        else:
            px, py = p
        return (infer_conv_size(dim_x, kx, sx, px),
                infer_conv_size(dim_y, ky, sy, py))

    def forward(self, x: torch.Tensor, return_full_list=False, clip_grad=False):
        '''Forward pass

        Args:
            x: Input.
            return_full_list: Optional, returns all layer outputs.

        Returns:
            torch.Tensor or list of torch.Tensor.

        '''

        def _clip_grad(v, min, max):
            v_tmp = v.expand_as(v)
            v_tmp.register_hook(lambda g: g.clamp(min, max))
            return v_tmp

        out = []
        for layer in self.layers:
            x = layer(x)
            if clip_grad:
                x = _clip_grad(x, -clip_grad, clip_grad)
            out.append(x)

        if not return_full_list:
            out = out[-1]

        return out


class FoldedConvnet(Convnet):
    '''Convnet with strided crop input.

    '''

    _supported_types = ('linear', 'conv', 'tconv', 'flatten', 'fold', 'unfold', None)

    def create_layers(self, shape, crop_size=8, layers=None):
        '''Creates layers

        Args:
            shape: Shape of input.
            crop_size: Size of crops
            layers: list of layer arguments.
        '''

        self.crop_size = crop_size
        self.layers, self.shapes = self.create_sequential(shape, layers=layers)

    def create_sequential(self, shape, layers=None):
        '''Creates a sequence of layers.

        Args:
            shape: Input shape.
            layers: list of layer arguments.

        Returns:
            nn.Sequential: a sequence of convolutional layers.

        '''

        self.final_size = None
        return super().create_sequential(shape, layers=layers)

    def handle_layer(self, block, shape, layer, layer_type):
        '''Handles the layer arguments and adds layer to the block.

        Args:
            block: nn.Sequential to add modules to.
            shape: Shape of the input.
            layer: Layer arguments.
            layer_type: Type of layer.

        Returns:
            tuple: Output shape.
        '''
        if layer_type == 'unfold':
            dim_x, dim_y, dim_out = shape
            self.final_size = 2 * (dim_x // self.crop_size) - 1
            block.add_module(layer_type, Unfold(dim_x, self.crop_size))
            shape = (self.crop_size, self.crop_size, dim_out)
        elif layer_type == 'fold':
            if self.final_size is None:
                raise ValueError('Cannot fold without unfolding first.')
            dim_out = shape[2]
            block.add_module(layer_type, Fold(self.final_size))
            shape = (self.final_size, self.final_size, dim_out)
        elif layer_type is None:
            pass
        else:
            shape = super().handle_layer(block, shape, layer, layer_type)

        return shape
