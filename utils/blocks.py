# flake8: noqa: E501
import torch
import torch.nn as nn
import torch.nn.functional as F



def weights_init(m, size=0.001):
    """
    Initialise the weights of a module. Does not change the default initialisation
    method of linear, conv2d, or conv2dtranspose layers.

    Parameters
    ----------
    m : torch.nn.Module
        Module to initialise
    
    size : float
        Standard deviation of the normal distribution to sample initial values from
        default: 0.001

    Returns
    -------
    None
    """

    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        # nn.init.trunc_normal_(m.weight, 1.0, size)

        # while torch.any(m.weight == 0.0):
        #     nn.init.trunc_normal_(m.weight, 1.0, size)

        if m.bias is not None:
            nn.init.trunc_normal_(m.bias, 0.0, size)

            while torch.any(m.bias == 0.0):
                nn.init.trunc_normal_(m.bias, 0.0, size)

    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)) and m.bias is not None:
        nn.init.trunc_normal_(m.bias, 0.0, size)

        while torch.any(m.bias == 0.0):
            nn.init.trunc_normal_(m.bias, 0.0, size)


class GaussianDropout2d(nn.Module):
    """
    Drop out channels of a 2D input with Gaussian noise.

    Parameters
    ----------
    p : float
        Probability of dropping a channel
        default: 0.5

    signal_to_noise : tuple
        Range of signal to noise ratios to use for the dropped channels. 0.0 is pure noise, 1.0 is pure signal.
        The amount of signal is randomly sampled from this range for each channel.
        If None, no signal is added to the dropped channels.
        default: (0.1, 0.9)
    """
    def __init__(self, p=0.5, signal_to_noise=(0.1, 0.9)):
        super(GaussianDropout2d, self).__init__()
        self.p = p
        self.signal_to_noise = signal_to_noise

    def forward(self, x):
        if self.training:
            batch_size, num_channels, height, width = x.size()

            # Create a mask of channels to drop
            mask = torch.rand(batch_size, num_channels, 1, 1, device=x.device) > self.p

            # If all channels are dropped, redraw the mask
            while torch.all(mask):
                mask = torch.rand(batch_size, num_channels, 1, 1, device=x.device) > self.p

            mean = x.mean([2, 3], keepdim=True).repeat(1, 1, height, width)
            std = x.std([2, 3], keepdim=True).repeat(1, 1, height, width)

            # Create the noise (Same mean and std as the input)
            noise = torch.normal(mean, torch.clamp(std, min=1e-6))

            if self.signal_to_noise is not None:
                signal_level = torch.rand(batch_size, num_channels, 1, 1, device=x.device) * (self.signal_to_noise[1] - self.signal_to_noise[0]) + self.signal_to_noise[0]
                adjusted_noise = noise * (1 - signal_level)
                adjusted_signal = x * signal_level

            # Combine the adjusted noise and signal
            return (adjusted_signal * mask) + (adjusted_noise * (~mask))
        
        return x
    

class GaussianDropout1d(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout1d, self).__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            batch_size, size = x.size()

            # Create a mask of channels to drop
            mask = torch.rand(batch_size, size, device=x.device) > self.p

            # If all channels are dropped, redraw the mask
            while torch.all(mask):
                mask = torch.rand(batch_size, size, device=x.device) > self.p

            mean = x.mean([1], keepdim=True).repeat(1, size)
            std = x.std([1], keepdim=True).repeat(1, size)

            # Create the noise (Same mean and std as the input)
            noise = torch.normal(mean, torch.clamp(std, min=1e-6))

            # Combine the adjusted noise and signal
            return (x * mask) + (noise * (~mask))
        
        return x


class RandomMask2D(nn.Module):
    """
    Randomly masks pixels of an image with zeros across all channels

    Parameters
    ----------
    p : float
        Probability of masking a pixel
        default: 0.5
    """
    def __init__(self, p=0.5):
        super(RandomMask2D, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.size(0), 1, x.size(2), x.size(3), device=x.device) > self.p

            return x * mask

        return x


class ScaleSkip2D(nn.Module):
    """
    Learnable channel-wise scale and bias for skip connections.
    
    Parameters
    ----------
    channels : int
        Number of channels in the input

    drop_y : float
        Probability of dropping a channel in the skip connection.
        Drops are replaces with Gaussian noise.

    signal_to_noise : tuple or None
        Range of signal to noise ratios to use for the dropped channels. 0.0 is pure noise, 1.0 is pure signal.
        The amount of signal is randomly sampled from this range for each channel.
        If None, no signal is added to the dropped channels.
        default: (0.1, 0.9)

    size : float
        Standard deviation of the normal distribution to sample inital values from
        default: 0.01
    """
    def __init__(self, channels, drop_y=None, signal_to_noise=(0.1, 0.9), size=0.01):
        super(ScaleSkip2D, self).__init__()
        self.channels = channels
        self.drop_y = drop_y
        self.size = size

        # Learnable scale and bias
        self.x_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.y_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.y_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        self.x_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))

        if self.drop_y is not None and self.drop_y > 0.0:
            self.drop_y = GaussianDropout2d(self.drop_y, signal_to_noise=signal_to_noise)
        else:
            self.drop_y = None

        self.set_weights()
        while (
            torch.any(self.x_skipscale == 0) 
            or torch.any(self.y_skipscale == 0) 
            or torch.any(self.y_skipbias == 0) 
            or torch.any(self.x_skipbias == 0)
        ):
            self.set_weights()

    def set_weights(self):
        nn.init.trunc_normal_(self.x_skipscale, 1.0, self.size)
        nn.init.trunc_normal_(self.y_skipscale, 1.0, self.size)
        nn.init.trunc_normal_(self.y_skipbias, 0.0, self.size)
        nn.init.trunc_normal_(self.x_skipbias, 0.0, self.size)

    def forward(self, x, y):
        x = (x * self.x_skipscale) + self.x_skipbias
        y = (y * self.y_skipscale) + self.y_skipbias

        if self.drop_y is not None:
            y = self.drop_y(y)

        return x + y


class ScaleSkip1D(nn.Module):
    """
    Learnable weight and bias for 1D skip connections.
    """
    def __init__(self, drop_y=None, size=0.01):
        super(ScaleSkip1D, self).__init__()

        self.size = size
        self.drop_y = drop_y

        # Learnable scale and bias
        self.x_skipscale = nn.Parameter(torch.ones(1, 1))
        self.y_skipscale = nn.Parameter(torch.ones(1, 1))
        self.y_skipbias = nn.Parameter(torch.zeros(1, 1))
        self.x_skipbias = nn.Parameter(torch.zeros(1, 1))

        self.set_weights()
        while torch.any(self.x_skipscale == 0) or torch.any(self.y_skipscale == 0) or torch.any(self.y_skipbias == 0) or torch.any(self.x_skipbias == 0):
            self.set_weights()

        if self.drop_y is not None and self.drop_y > 0.0:
            self.drop_y = GaussianDropout1d(self.drop_y)
        else:
            self.drop_y = None

    def set_weights(self):
        nn.init.trunc_normal_(self.x_skipscale, 1.0, self.size)
        nn.init.trunc_normal_(self.y_skipscale, 1.0, self.size)
        nn.init.trunc_normal_(self.y_skipbias, 0.0, self.size)
        nn.init.trunc_normal_(self.x_skipbias, 0.0, self.size)

    def forward(self, x, y):
        x = (x * self.x_skipscale) + self.x_skipbias
        y = (y * self.y_skipscale) + self.y_skipbias

        if self.drop_y is not None:
            y = self.drop_y(y)

        return x + y


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(1, channels // self.reduction), bias=False),
            nn.GELU(),
            nn.Linear(max(1, channels // self.reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)

        return x * y.expand_as(x)


class CNNBlock(nn.Module):
    """
    This is a standard CNN block with a 1x1 convolutional matcher for the skip connection.
    It adds a learnable scale and bias to the skip connection.

    Parameters
    ----------
    channels_in : int
        Number of channels in the input

    channels_out : int or None
        Number of channels in the output. If None, the number of channels is unchanged.
        default: None

    group_size : int
        Number of groups for the 3x3 convolution.
        default: 1

    activation : torch.nn.Module
        Activation function to use after the first convolution.
        default: torch.nn.GELU()

    activation_out : torch.nn.Module or None
        Activation function to use after the last convolution. If None, the same activation as the first convolution is used.
        default: None

    chw : tuple or None
        Height and width of the input. If None, batch norm is used instead of layer norm.
        default: None
    """
    def __init__(
        self,
        channels_in,
        channels_out=None,
        chw=None,
        group_size=1,
        activation=nn.GELU(),
        activation_out=None,
        residual=True,
        reduction=1,
    ):
        super().__init__()

        assert chw is not None, "chw must be specified"

        self.channels_in = channels_in
        self.channels_out = channels_in if channels_out is None else channels_out
        self.channels_internal = self.channels_out // reduction
        self.chw = chw
        self.group_size = group_size
        self.activation = activation
        self.activation_out = activation if activation_out is None else activation_out
        self.residual = residual
        self.reduction = reduction
        self.squeeze = SE_Block(self.channels_out, 16)

        self.matcher = nn.Conv2d(self.channels_in, self.channels_out, 1, padding=0, bias=False) if self.channels_in != self.channels_out else None

        self.norm1 = nn.LayerNorm([self.channels_internal, self.chw[1], self.chw[2]])
        self.norm2 = nn.LayerNorm([self.channels_internal, self.chw[1], self.chw[2]])

        self.conv1 = nn.Conv2d(self.channels_in, self.channels_internal, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.channels_internal, self.channels_internal, 3, padding=1, groups=self.group_size, bias=False, padding_mode="replicate")
        self.conv3 = nn.Conv2d(self.channels_internal, self.channels_out, 1, padding=0, bias=True)

        self.scaler = ScaleSkip2D(self.channels_out) if self.residual else None


    def forward(self, x):
        identity = x if self.matcher is None else self.matcher(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.squeeze(x)

        if self.residual:
            x = self.scaler(x, identity)

        x = self.activation_out(x)

        return x


class GlobalBlock(nn.Module):
    """
    Global Block for the paper `'Global Context Dynamic-CNNs (Fibaek et al., 2024)'`

    Parameters
    ----------
    in_channels : int
        Number of input channels

    out_channels : int or None
        Number of output channels. If None, the number of channels is unchanged.
        default: None

    kernel_size : int
        Size of the second convolutional kernel.
        default: 3

    patch_size : int
        Size of the patches to split the image into.
        default: 16

    chw : tuple
        Height and width of the input. Must be divisible by patch_size.
        default: None

    activation : torch.nn.Module
        Activation function to use after the first convolution.
        default: torch.nn.GELU()

    activation_out : torch.nn.Module or None
        Activation function to use after the last convolution. If None, the same activation as the first convolution is used.
        default: None

    reduction : int
        Reduction factor for the internal channels.
        default: 1 (no reduction)

    residual : bool
        Whether to use a residual connection.
        default: True

    patch_dim : int
        Dimension of the patch embeddings.
        default: 512

    projector : torch.nn.Module or None
        Projector to use for the patch embeddings. If None, a new projector is created.
        ```python
        self.projector = nn.Sequential(
            nn.Linear(self.in_channels * (self.patch_size ** 2), self.patch_dim),
            nn.LayerNorm(self.patch_dim),
            nn.GELU(),
        )
        ```
        default: None        
    """
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        patch_size=16,
        chw=None,
        activation=nn.GELU(),
        activation_out=None,
        reduction=1,
        residual=True,
        patch_dim=512,
        shared_context=32,
        num_heads=8,
        projector=None,
    ):
        super(GlobalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.chw = chw
        self.activation = activation
        self.activation_out = activation_out if activation_out is not None else activation
        self.reduction = reduction
        self.residual = residual
        self.patch_dim = patch_dim
        self.shared_context = shared_context
        self.num_heads = num_heads
        self.projector = projector       

        assert chw is not None, "chw must be specified"
        assert chw[1] == chw[2], "chw must be square"
        assert chw[1] % patch_size == 0, "chw must be divisible by patch_size"
        assert chw[1] >= patch_size, "patch_size must be greater than or equal to chw"

        self.num_patches_height = self.chw[1] // self.patch_size
        self.num_patches_width = self.chw[2] // self.patch_size
        self.num_patches = self.num_patches_height * self.num_patches_width

        self.internal_channels = self.out_channels // self.reduction

        self.latent_1x1 = self.out_channels * self.internal_channels + self.internal_channels
        self.latent_3x3 = self.internal_channels * self.internal_channels * kernel_size * kernel_size + self.internal_channels
        self.latent_1x1_out = self.internal_channels * self.out_channels + self.out_channels
        self.context_size = (self.patch_dim * self.num_patches) + (self.shared_context * 3)

        self.projector = nn.Linear(self.out_channels * (self.patch_size ** 2), self.patch_dim) if projector is None else projector

        self.conv_1x1 = nn.Linear(self.context_size, self.latent_1x1 + self.shared_context)
        self.conv_3x3 = nn.Linear(self.context_size, self.latent_3x3 + self.shared_context)
        self.conv_1x1_out = nn.Linear(self.context_size, self.latent_1x1_out + self.shared_context)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.patch_dim, num_heads=self.num_heads, batch_first=True, add_zero_attn=True, add_bias_kv=True)

        self.scaler = ScaleSkip2D(self.out_channels)
        self.scaler_ctx = ScaleSkip1D()

        self.pos_embed = self.posemb_sincos_2d(self.num_patches_height, self.num_patches_width, self.patch_dim)

        # So many normalisations - so little time
        self.norm_input = nn.LayerNorm([self.out_channels, self.chw[1], self.chw[2]])
        self.norm_patches1 = nn.LayerNorm([self.num_patches, self.patch_dim])
        self.norm_patches2 = nn.LayerNorm([self.num_patches, self.patch_dim])
        self.context_norm_1x1 = nn.LayerNorm(self.context_size)
        self.context_norm_3x3 = nn.LayerNorm(self.context_size)
        self.context_norm_1x1_out = nn.LayerNorm(self.context_size)
        self.context_shared_norm = nn.LayerNorm((self.shared_context * 3))
        self.cnn_norm_1x1 = nn.LayerNorm([self.internal_channels, self.chw[1], self.chw[2]])
        self.cnn_norm_3x3 = nn.LayerNorm([self.internal_channels, self.chw[1], self.chw[2]])

        if self.in_channels == self.out_channels:
            self.matcher = nn.Identity()
        else:
            self.matcher = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)

        self.apply(weights_init)


    def patchify_batch(self, tensor):
        """ 
        Split a batch of images into patches.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Batch of images to split into patches
            Shape: `(B, C, H, W)`

        Returns
        -------
        torch.Tensor
            Batch of patches
            Shape: `(B, num_patches, C * (patch_size ** 2))`
        """
        B, C, _H, _W = tensor.shape

        reshaped = tensor.reshape(B, C, self.num_patches_height, self.patch_size, self.num_patches_width, self.patch_size)
        transposed = reshaped.permute(0, 2, 4, 1, 3, 5)
        patches = transposed.reshape(B, self.num_patches, C * self.patch_size * self.patch_size)

        return patches


    def convolve(self, x, context, in_channels, out_channels, size=3):
        """
        Perform a convolution with a learned context (kernel) vector.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            Shape: `(B, C, H, W)`

        context : torch.Tensor
            Context vector, reshaped into a kernel
            Shape: `(B, out_channels, in_channels, size, size)`

        in_channels : int
            Number of input channels

        out_channels : int
            Number of output channels

        size : int
            Size of the kernel
            default: 3

        Returns
        -------
        torch.Tensor
            Convolved output tensor
            Shape: `(B, out_channels, H, W)`
        """
        batch_size = x.size(0)

        _kernel, _bias = torch.split(context, [context.size(1) - out_channels, out_channels], dim=1)
        kernel = _kernel.reshape(batch_size * out_channels, in_channels, size, size)
        bias = _bias.reshape(batch_size * out_channels)

        x = x.reshape(1, batch_size * in_channels, x.shape[2], x.shape[3])
        if size != 1:
            x = F.pad(x, (self.kernel_size // 2,) * 4) # pylint: disable=not-callable

        x = F.conv2d(x, kernel, groups=batch_size, bias=bias) # pylint: disable=not-callable
        x = x.reshape(batch_size, out_channels, x.shape[2], x.shape[3])

        return x
    

    def posemb_sincos_2d(self, h, w, dim, temperature: int = 10000, dtype = torch.float32):
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature ** omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]

        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

        return pe.type(dtype)


    def forward_stem(self, x):
        x = self.patchify_batch(x) # shape: (B, num_patches, C * (patch_size ** 2))
        x = self.projector(x) # shape: (B, num_patches, patch_dim)
        x = self.norm_patches1(x)
        attn, _ = self.multihead_attn(x, x, x) # self-attention
        x = x + attn
        x = self.norm_patches2(x)
        x = x + self.pos_embed.to(x.device)
        x = x.reshape(x.shape[0], -1) # shape: (B, num_patches * patch_dim)

        return x


    def forward(self, x, ctx_p=None):
        identity = x if self.in_channels == self.out_channels else self.matcher(x) # shape: (B, C, H, W)
        x = self.norm_input(identity)

        ctx_p = torch.zeros((x.shape[0], self.shared_context * 3), device=x.device) if ctx_p is None else ctx_p # shape: (B, shared_context)
        prev_1x1, prev_3x3, prev_1x1_out = ctx_p.split([self.shared_context, self.shared_context, self.shared_context], dim=1)
        
        embedded_patches = self.forward_stem(x) # shape: (B, num_patches * patch_dim)

        # 1x1 Convolution with global context
        inputs = self.context_norm_1x1(torch.cat([embedded_patches, prev_1x1, prev_3x3, prev_1x1_out], dim=1))
        combined_context = self.conv_1x1(inputs)
        ctx, ctx_1x1 = combined_context.split([self.latent_1x1, self.shared_context], dim=1)
        x = self.convolve(x, ctx, self.out_channels, self.internal_channels, size=1)
        x = self.cnn_norm_1x1(x)

        # 3x3 Convolution with global context
        inputs = self.context_norm_3x3(torch.cat([embedded_patches, ctx_1x1, prev_3x3, prev_1x1_out], dim=1))
        combined_context = self.conv_3x3(inputs)
        ctx, ctx_3x3 = combined_context.split([self.latent_3x3, self.shared_context], dim=1)
        x = self.convolve(x, ctx, self.internal_channels, self.internal_channels, size=self.kernel_size)
        x = self.cnn_norm_3x3(x)

        # 1x1 Convolution with global context
        inputs = self.context_norm_1x1_out(torch.cat([embedded_patches, ctx_1x1, ctx_3x3, prev_1x1_out], dim=1))
        combined_context = self.conv_1x1_out(inputs)
        ctx, ctx_1x1_out = combined_context.split([self.latent_1x1_out, self.shared_context], dim=1)
        x = self.convolve(x, ctx, self.internal_channels, self.out_channels, size=1)

        # Merge contexts
        ctx_o = self.context_shared_norm(torch.cat([ctx_1x1, ctx_3x3, ctx_1x1_out], dim=1))

        # Learned skip-connection
        x = self.scaler(x, identity) if self.residual else x
        ctx_o = self.scaler_ctx(ctx_o, ctx_p) if self.residual else ctx_o

        # Activation
        x = self.activation_out(x)
        ctx_o = self.activation_out(ctx_o)

        return x, ctx_o
