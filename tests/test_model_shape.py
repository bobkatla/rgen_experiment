import torch
from rgen.core.model import UNet32

def test_unet32_shapes():
    b = 4
    x = torch.randn(b, 3, 32, 32)
    t = torch.randint(0, 1000, (b,), dtype=torch.long)
    y = torch.randint(0, 10, (b,), dtype=torch.long)
    m = UNet32(num_classes=10, cond_null_id=10)
    out = m(x, t, y)
    assert out.shape == (b, 6, 32, 32)  # predict_sigma=True -> 2*C channels
