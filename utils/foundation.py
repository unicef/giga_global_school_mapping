import torch.nn as nn
from blocks import CNNBlock, ScaleSkip2D, RandomMask2D



class FoundationEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim=3,
        depths=None,
        dims=None,
        img_size=64,
        latent_dim=512,
        activation=nn.GELU(),
    ):
        super().__init__()

        self.depths = depths
        self.dims = dims
        self.input_dim = input_dim
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.steps = 1
        self.sizes = [img_size]
        self.activation = activation

        for i in range(len(self.depths) - 1):
            half = self.sizes[-1] // 2
            self.sizes.append(half)
            self.steps += 1

        self.linear_dim = int(((img_size // (2 ** (self.steps - 1))) ** 2) * self.dims[-1])

        assert len(self.depths) == self.steps, "Invalid depths"
        assert len(self.dims) == self.steps, "Invalid dims"
        assert self.depths is not None, "Invalid depths"
        assert self.dims is not None, "Invalid dims"
        assert self.steps == len(self.dims), "Invalid dims"

        self.downsample = nn.ModuleList()
        for i in range(self.steps - 1):
            self.downsample.append(nn.Sequential(
                nn.Conv2d(self.dims[i], self.dims[i + 1], 1, padding=0),
                nn.MaxPool2d(2, stride=2),
            ))

        self.block_scalers = nn.ModuleList()
        for i in range(self.steps):
            self.block_scalers.append(ScaleSkip2D(self.dims[i]))

        self.blocks_down = nn.ModuleList()
        for i in range(self.steps):
            self.blocks_down.append(nn.ModuleList())
            for _ in range(self.depths[i]):
                self.blocks_down[i].append(
                    CNNBlock(self.dims[i], chw=[self.dims[i], self.sizes[i], self.sizes[i]], activation=self.activation)
                )

        self.linear_encode = nn.Sequential(
            nn.LayerNorm(self.linear_dim),
            self.activation,
            nn.Linear(self.linear_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

        self.head_climate_l1 = nn.Sequential(
            nn.Linear(self.latent_dim, 5),
        )

        self.head_climate_l2 = nn.Sequential(
            nn.Linear(self.latent_dim, 12),
        )

        self.head_zoom = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
            nn.ReLU6(),
        )

        self.head_buildings = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid(),
        )

        self.head_landcover = nn.Sequential(
            nn.Linear(self.latent_dim, 11),
        )

        self.head_coords = nn.Sequential(
            nn.Linear(self.latent_dim, 4),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        skips = []

        for i in range(self.steps):
            pre_block = x
            for j in range(self.depths[i]):
                block = self.blocks_down[i][j]
                x = block(x)

            if len(self.blocks_down[i]) > 1:
                x = self.block_scalers[i](x, pre_block)

            skips.append(x)

            if i < self.steps - 1:
                x = self.downsample[i](x)

        flat = x.reshape(-1, self.linear_dim)
        embeddings = self.linear_encode(flat)
        out_landcover = self.head_landcover(embeddings) # 10
        out_coords = self.head_coords(embeddings) # 4
        out_zoom = self.head_zoom(embeddings)
        out_buildings = self.head_buildings(embeddings)
        out_climate_l1 = self.head_climate_l1(embeddings)
        out_climate_l2 = self.head_climate_l2(embeddings)

        return (
            embeddings,
            skips,
            (
                out_landcover,
                out_coords,
                out_zoom,
                out_buildings,
                out_climate_l1,
                out_climate_l2,
            )
        )


class FoundationDecoder(nn.Module):
    def __init__(
        self,
        *,
        depths=None,
        dims=None,
        img_size=64,
        latent_dim=512,
        dropout=None,
        activation=nn.GELU(),
    ):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.steps = 1
        self.sizes = [img_size]
        self.dropout = dropout
        self.activation = activation

        for i in range(len(self.depths) - 1):
            half = self.sizes[-1] // 2
            self.sizes.append(half)
            self.steps += 1

        self.sizes = self.sizes[::-1]
        self.linear_dim = int(((img_size // (2 ** (self.steps - 1))) ** 2) * self.dims[0])

        if self.dropout is None:
            self.dropout = [0.0] * self.steps
        elif isinstance(self.dropout, (int, float)):
            self.dropout = [self.dropout] * self.steps

        assert len(self.depths) == self.steps, "Invalid depths"
        assert len(self.dims) == self.steps, "Invalid dims"
        assert len(self.dropout) == self.steps, "Invalid dropout"
        assert self.depths is not None, "Invalid depths"
        assert self.dims is not None, "Invalid dims"
        assert self.dropout is not None, "Invalid dropout"

        self.linear_decode = nn.Linear(self.latent_dim, self.linear_dim)

        self.latent_norm = nn.LayerNorm([self.dims[0], self.img_size // (2 ** (self.steps - 1)), self.img_size // (2 ** (self.steps - 1))])
        self.prehead_norm = nn.LayerNorm([self.dims[-1], self.sizes[-1], self.sizes[-1]])

        self.skip_scalers = nn.ModuleList()
        self.block_scalers = nn.ModuleList()
        self.maskers = nn.ModuleList()
        for i in range(self.steps):
            self.skip_scalers.append(ScaleSkip2D(self.dims[i], drop_y=self.dropout[i], signal_to_noise=(0.1, 0.9)))
            self.block_scalers.append(ScaleSkip2D(self.dims[i]))
            self.maskers.append(RandomMask2D(p=0.5))

        self.blocks_up = nn.ModuleList()
        for i in range(self.steps):
            self.blocks_up.append(nn.ModuleList())
            for _ in range(self.depths[i]):
                self.blocks_up[i].append(
                    CNNBlock(self.dims[i], chw=[self.dims[i], self.sizes[i], self.sizes[i]], activation=self.activation)
                )

        self.upsamplers = nn.ModuleList()
        for i in range(self.steps - 1):
            self.upsamplers.append(nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(self.dims[i], self.dims[i + 1], 3, padding=1, bias=False, padding_mode='replicate'),
                nn.LayerNorm([self.dims[i + 1], self.sizes[i + 1], self.sizes[i + 1]]),
                self.activation,
            ))

    def forward(self, x, skips):
        x = self.linear_decode(x)
        x = x.reshape(-1, self.dims[0], self.img_size // (2 ** (self.steps - 1)), self.img_size // (2 ** (self.steps - 1)))
        x = self.latent_norm(x)

        for i in range(self.steps):
            skip_x = skips[-(i + 1)]
            skip_x = self.maskers[i](skip_x) # mask random pixels

            x = self.skip_scalers[i](x, skip_x)

            pre_block = x
            for block in self.blocks_up[i]:
                x = block(x)

            if len(self.blocks_up[i]) > 1:
                x = self.block_scalers[i](x, pre_block)

            if i < self.steps - 1:
                x = self.upsamplers[i](x)

        x = self.prehead_norm(x)

        return x


class Foundation(nn.Module):
    def __init__(
        self,
        *,
        input_dim=3,
        output_dim=None,
        depths=None,
        dims=None,
        img_size=64,
        latent_dim=512,
        dropout=None,
        activation=nn.GELU(),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.depths = depths
        self.dims = dims
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.activation = activation

        self.stem = CNNBlock(
            input_dim,
            dims[0],
            chw=[input_dim, img_size, img_size],
            activation=self.activation,
        )

        self.encoder = FoundationEncoder(
            input_dim=dims[0],
            depths=depths,
            dims=dims,
            img_size=img_size,
            latent_dim=latent_dim,
            activation=self.activation,
        )

        self.decoder = FoundationDecoder(
            depths=depths[::-1],
            dims=dims[::-1],
            img_size=img_size,
            latent_dim=latent_dim,
            dropout=dropout,
            activation=self.activation,
        )

        self.head = CNNBlock(
            self.dims[0],
            self.output_dim,
            chw=[self.output_dim, self.img_size, self.img_size],
            activation=self.activation,
            activation_out=nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.stem(x)
        embeddings, skips, (out_landcover, out_coords, out_zoom, out_buildings, out_climate_l1, out_climate_l2) = self.encoder(x)
        decoded = self.decoder(embeddings, skips)
        reconstruction = self.head(decoded)

        return reconstruction, embeddings, decoded, (out_landcover, out_coords, out_zoom, out_buildings, out_climate_l1, out_climate_l2)
