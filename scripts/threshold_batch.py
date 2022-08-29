import torch
import numpy as np
from einops import rearrange
from PIL import Image

dynamic_thresholding_percentile=0.9995

def right_pad_dims_to(x, t):
  padding_dims = x.ndim - t.ndim
  if padding_dims <= 0:
    return t
  return t.view(*t.shape, *((1,) * padding_dims))

device = torch.device('cpu')
x_samples = torch.load('x.pt', map_location=device)

s = torch.quantile(
  rearrange(x_samples, 'a b ... -> a b (...)').abs(),
  dynamic_thresholding_percentile,
  dim = 2
)
s.clamp_(min = 1.)
s = right_pad_dims_to(x_samples, s)
x_samples = x_samples.clamp(-s, s) / s
x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
x_samples = x_samples.cpu().permute(0, 2, 3, 1)
x_checked_image = x_samples.numpy()

x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

for ix, x_sample in enumerate(x_checked_image_torch):
  x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
  img = Image.fromarray(x_sample.astype(np.uint8))
  img.save(f'xbatch.{ix}.png')