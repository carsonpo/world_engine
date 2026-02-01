import mlx.core as mx
import mlx.nn as nn


"""
MLX version of the autoencoder.
WARNING:
- Always assumes scale=1, shift=0
- Weights should be pre-baked (weight_norm removed)
- Uses NHWC format throughout (MLX native)
"""


def pixel_shuffle_nhwc(x: mx.array, upscale_factor: int) -> mx.array:
    """NHWC pixel shuffle: rearrange (N, H, W, C*r^2) -> (N, H*r, W*r, C)

    Matches PyTorch's pixel_shuffle channel ordering where input channels
    are arranged as [C, r, r] flattened.
    """
    b, h, w, c = x.shape
    r = upscale_factor
    oc = c // (r * r)
    # Input channels are ordered as [oc, r, r] flattened (to match PyTorch)
    x = x.reshape(b, h, w, oc, r, r)
    # Rearrange to interleave spatial dimensions: (b, h, r, w, r, oc)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    x = x.reshape(b, h * r, w * r, oc)
    return x


def pixel_unshuffle_nhwc(x: mx.array, downscale_factor: int) -> mx.array:
    """NHWC pixel unshuffle: rearrange (N, H*r, W*r, C) -> (N, H, W, C*r^2)

    Matches PyTorch's pixel_unshuffle channel ordering where output channels
    are arranged as [C, r, r] flattened.
    """
    b, h, w, c = x.shape
    r = downscale_factor
    oh, ow = h // r, w // r
    # Reshape to separate the r factors from H and W
    x = x.reshape(b, oh, r, ow, r, c)
    # Rearrange to (b, oh, ow, c, r, r) to match PyTorch's channel ordering
    x = mx.transpose(x, (0, 1, 3, 5, 2, 4))
    x = x.reshape(b, oh, ow, c * r * r)
    return x


def _cubic_interp_1d(
    x0: mx.array, x1: mx.array, x2: mx.array, x3: mx.array, t: mx.array
) -> mx.array:
    """Cubic interpolation along one dimension.

    Uses the same formula as PyTorch's upsample_bicubic2d.
    x0, x1, x2, x3 are the 4 sample values, t is the fractional position in [0, 1).
    """
    a = -0.75  # PyTorch uses -0.75 for bicubic
    coeffs_0 = ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a
    coeffs_1 = ((a + 2) * t - (a + 3)) * t * t + 1
    coeffs_2 = ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1
    coeffs_3 = ((a * (2 - t) - 5 * a) * (2 - t) + 8 * a) * (2 - t) - 4 * a

    return x0 * coeffs_0 + x1 * coeffs_1 + x2 * coeffs_2 + x3 * coeffs_3


def interpolate_bicubic_real(
    x: mx.array, size: tuple = None, scale_factor: float = None
) -> mx.array:
    """Bicubic interpolation for NHWC tensors.

    Matches PyTorch's F.interpolate with mode='bicubic', align_corners=False.
    """
    b, h, w, c = x.shape

    if size is not None:
        new_h, new_w = size
    elif scale_factor is not None:
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
    else:
        raise ValueError("Either size or scale_factor must be provided")

    # Coordinate mapping with align_corners=False
    # Maps output pixel centers to input coordinates
    y_coords = (mx.arange(new_h).astype(mx.float32) + 0.5) * h / new_h - 0.5
    x_coords = (mx.arange(new_w).astype(mx.float32) + 0.5) * w / new_w - 0.5

    # Get the integer part (floor) - this is the index of x1 in the 4-tap filter
    y_floor = mx.floor(y_coords).astype(mx.int32)
    x_floor = mx.floor(x_coords).astype(mx.int32)

    # Fractional part for interpolation weights
    fy = y_coords - y_floor.astype(mx.float32)
    fx = x_coords - x_floor.astype(mx.float32)

    # Compute the 4 y and x indices, clamped to valid range
    # PyTorch clamps indices to [0, size-1]
    y0 = mx.clip(y_floor - 1, 0, h - 1)
    y1 = mx.clip(y_floor, 0, h - 1)
    y2 = mx.clip(y_floor + 1, 0, h - 1)
    y3 = mx.clip(y_floor + 2, 0, h - 1)

    x0 = mx.clip(x_floor - 1, 0, w - 1)
    x1 = mx.clip(x_floor, 0, w - 1)
    x2 = mx.clip(x_floor + 1, 0, w - 1)
    x3 = mx.clip(x_floor + 2, 0, w - 1)

    # Gather rows for each of the 4 y positions
    # Shape of each: [b, new_h, w, c]
    row0 = x[:, y0]
    row1 = x[:, y1]
    row2 = x[:, y2]
    row3 = x[:, y3]

    # For each row, gather the 4 x positions and interpolate horizontally
    # fx: [new_w] -> [1, 1, new_w, 1] for broadcasting
    fx_bc = fx[None, None, :, None]

    def interp_row_x(row):
        # row: [b, new_h, w, c]
        # Gather 4 x positions: [b, new_h, new_w, c] each
        p0 = row[:, :, x0]
        p1 = row[:, :, x1]
        p2 = row[:, :, x2]
        p3 = row[:, :, x3]
        return _cubic_interp_1d(p0, p1, p2, p3, fx_bc)

    # Interpolate each row horizontally
    h_row0 = interp_row_x(row0)  # [b, new_h, new_w, c]
    h_row1 = interp_row_x(row1)
    h_row2 = interp_row_x(row2)
    h_row3 = interp_row_x(row3)

    # Interpolate vertically
    # fy: [new_h] -> [1, new_h, 1, 1] for broadcasting
    fy_bc = fy[None, :, None, None]

    result = _cubic_interp_1d(h_row0, h_row1, h_row2, h_row3, fy_bc)

    return result


def interpolate_bicubic_dummy(
    x: mx.array, size: tuple = None, scale_factor: float = None
) -> mx.array:
    """Dummy interpolation - just nearest neighbor to test if bicubic is the bottleneck."""
    b, h, w, c = x.shape

    if size is not None:
        new_h, new_w = size
    elif scale_factor is not None:
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
    else:
        raise ValueError("Either size or scale_factor must be provided")

    # Simple nearest neighbor - just index mapping, no computation
    y_idx = (mx.arange(new_h) * h // new_h).astype(mx.int32)
    x_idx = (mx.arange(new_w) * w // new_w).astype(mx.int32)

    # Single gather operation
    result = x[:, y_idx][:, :, x_idx]
    return result


# Toggle this to switch between real and dummy interpolation
USE_DUMMY_INTERPOLATE = True


def interpolate_bicubic(
    x: mx.array, size: tuple = None, scale_factor: float = None
) -> mx.array:
    if USE_DUMMY_INTERPOLATE:
        return interpolate_bicubic_dummy(x, size, scale_factor)
    else:
        return interpolate_bicubic_real(x, size, scale_factor)


def interpolate_bilinear(
    x: mx.array, size: tuple = None, scale_factor: float = None
) -> mx.array:
    """Bilinear interpolation for NHWC tensors.

    Matches PyTorch's F.interpolate with mode='bilinear', align_corners=False.
    """
    b, h, w, c = x.shape

    if size is not None:
        new_h, new_w = size
    elif scale_factor is not None:
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
    else:
        raise ValueError("Either size or scale_factor must be provided")

    # Create sampling grid with align_corners=False
    # Formula: src = (dst + 0.5) * src_size / dst_size - 0.5
    y_coords = (mx.arange(new_h).astype(mx.float32) + 0.5) * h / new_h - 0.5
    x_coords = (mx.arange(new_w).astype(mx.float32) + 0.5) * w / new_w - 0.5

    # Clamp to valid range
    y_coords = mx.clip(y_coords, 0, h - 1)
    x_coords = mx.clip(x_coords, 0, w - 1)

    # Get integer and fractional parts
    y0 = mx.floor(y_coords).astype(mx.int32)
    x0 = mx.floor(x_coords).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, h - 1)
    x1 = mx.minimum(x0 + 1, w - 1)

    fy = (y_coords - y0.astype(x.dtype))[:, None, None]  # [new_h, 1, 1]
    fx = (x_coords - x0.astype(x.dtype))[None, :, None]  # [1, new_w, 1]

    def gather_2d(arr, yi, xi):
        # arr: [b, h, w, c], yi: [new_h], xi: [new_w]
        return arr[:, yi][:, :, xi]

    top_left = gather_2d(x, y0, x0)
    top_right = gather_2d(x, y0, x1)
    bottom_left = gather_2d(x, y1, x0)
    bottom_right = gather_2d(x, y1, x1)

    top = top_left * (1 - fx) + top_right * fx
    bottom = bottom_left * (1 - fx) + bottom_right * fx
    result = top * (1 - fy) + bottom * fy

    return result


# === General Blocks ===


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()

        hidden = 2 * ch
        n_grps = max(1, hidden // 16)

        self.conv1 = nn.Conv2d(ch, hidden, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(
            hidden, hidden, kernel_size=3, stride=1, padding=1, groups=n_grps
        )
        self.conv3 = nn.Conv2d(
            hidden, ch, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(x)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.act2(h)
        h = self.conv3(h)
        return x + h


# === Encoder ===


class LandscapeToSquare(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = interpolate_bicubic(x, size=(512, 512))
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = nn.Conv2d(
            ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = interpolate_bicubic(x, scale_factor=0.5)
        x = self.proj(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, num_res: int = 1):
        super().__init__()

        self.down = Downsample(ch_in, ch_out)
        self.blocks = [ResBlock(ch_in) for _ in range(num_res)]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        x = self.down(x)
        return x


class SpaceToChannel(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = nn.Conv2d(ch_in, ch_out // 4, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        x = pixel_unshuffle_nhwc(x, 2)
        return x


class ChannelAverage(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.grps = ch_in // ch_out
        self.scale = (self.grps) ** 0.5

    def __call__(self, x: mx.array) -> mx.array:
        res = x  # NHWC: [b, h, w, c]
        x = self.proj(x)

        b, h, w, c = res.shape
        res = res.reshape(b, h, w, self.grps, c // self.grps)
        res = res.mean(axis=3) * self.scale

        return res + x


# === Decoder ===


class SquareToLandscape(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        x = interpolate_bicubic(x, size=(360, 640))
        return x


class Upsample(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        if ch_in == ch_out:
            self.proj = None
        else:
            self.proj = nn.Conv2d(
                ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False
            )

    def __call__(self, x: mx.array) -> mx.array:
        if self.proj is not None:
            x = self.proj(x)
        x = interpolate_bicubic(x, scale_factor=2.0)
        return x


class UpBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, num_res: int = 1):
        super().__init__()

        self.up = Upsample(ch_in, ch_out)
        self.blocks = [ResBlock(ch_out) for _ in range(num_res)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.up(x)
        for block in self.blocks:
            x = block(x)
        return x


class ChannelToSpace(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = nn.Conv2d(ch_in, ch_out * 4, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        x = pixel_shuffle_nhwc(x, 2)
        return x


class ChannelDuplication(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.reps = ch_out // ch_in
        self.scale = (self.reps) ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        res = x  # NHWC: [b, h, w, c]
        x = self.proj(x)

        b, h, w, c = res.shape
        # Expand and reshape to match PyTorch's channel ordering [c, reps]
        res = mx.expand_dims(res, axis=4)  # [b, h, w, c, 1]
        res = mx.broadcast_to(res, (b, h, w, c, self.reps))  # [b, h, w, c, reps]
        res = res.reshape(b, h, w, c * self.reps)  # [c, reps] ordering matches PyTorch
        res = res * self.scale

        return res + x


# === Main AE ===


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv_in = LandscapeToSquare(config.channels, config.ch_0)

        blocks = []
        residuals = []

        ch = config.ch_0
        for block_count in config.encoder_blocks_per_stage:
            next_ch = min(ch * 2, config.ch_max)

            blocks.append(DownBlock(ch, next_ch, block_count))
            residuals.append(SpaceToChannel(ch, next_ch))

            ch = next_ch

        self.blocks = blocks
        self.residuals = residuals
        self.conv_out = ChannelAverage(ch, config.latent_channels)

        self.skip_logvar = bool(getattr(config, "skip_logvar", False))
        if not self.skip_logvar:
            self.conv_out_logvar = nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block, residual in zip(self.blocks, self.residuals):
            x = block(x) + residual(x)
        return self.conv_out(x)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv_in = ChannelDuplication(config.latent_channels, config.ch_max)

        blocks = []
        residuals = []

        ch = config.ch_0
        for block_count in reversed(config.decoder_blocks_per_stage):
            next_ch = min(ch * 2, config.ch_max)

            blocks.append(UpBlock(next_ch, ch, block_count))
            residuals.append(ChannelToSpace(next_ch, ch))

            ch = next_ch

        self.blocks = list(reversed(blocks))
        self.residuals = list(reversed(residuals))

        self.act_out = nn.SiLU()
        self.conv_out = SquareToLandscape(config.ch_0, config.channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block, residual in zip(self.blocks, self.residuals):
            x = block(x) + residual(x)
        x = self.act_out(x)
        return self.conv_out(x)


class AutoEncoder(nn.Module):
    def __init__(self, encoder_config, decoder_config=None):
        super().__init__()

        if decoder_config is None:
            decoder_config = encoder_config

        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)

    def __call__(self, x: mx.array) -> mx.array:
        return self.decoder(self.encoder(x))


def convert_torch_weights_to_mlx(torch_state_dict: dict) -> dict:
    """
    Convert PyTorch state dict to MLX format.
    PyTorch Conv2d weight: (out_channels, in_channels, kH, kW)
    MLX Conv2d weight: (out_channels, kH, kW, in_channels)
    """
    import numpy as np

    mlx_weights = {}
    for key, value in torch_state_dict.items():
        arr = value.numpy() if hasattr(value, "numpy") else value

        # Check if this is a conv weight (4D tensor)
        if arr.ndim == 4:
            # Transpose from OIHW to OHWI
            arr = np.transpose(arr, (0, 2, 3, 1))

        mlx_weights[key] = mx.array(arr)

    return mlx_weights


if __name__ == "__main__":
    pass
