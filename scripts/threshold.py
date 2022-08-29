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
for x_sample in x_samples:
  s = torch.quantile(
    rearrange(x_sample, 'b ... -> b (...)').abs(),
    dynamic_thresholding_percentile,
    dim = -1
  )
  s.clamp_(min = 1.)
  s = right_pad_dims_to(x_sample, s)
  x_sample = x_sample.clamp(-s, s) / s
  x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
  x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
  img = Image.fromarray(x_sample.astype(np.uint8))
  img.save('x.png')

# torch.min(x_samples)
# tensor(-1.3103)
# torch.max(x_samples)
# tensor(1.3893)
# x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
# print('clamped:')
# print(x_samples)
# x_samples = x_samples.cpu().permute(0, 2, 3, 1)
# print('permuted:')
# print(x_samples)
# x_checked_image = x_samples.numpy()

# x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

# for x_sample in x_checked_image_torch:
#   x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
#   img = Image.fromarray(x_sample.astype(np.uint8))
#   img.save('x.png')