# sanity_denoise.py
import torch
import json
from torchvision import datasets, transforms
from rgen.core.model import UNet32
from rgen.core.diffusion import GaussianDiffusion, DiffusionConfig

# 1) load EMA ckpt
ckpt = torch.load("output/cifar10_unet_eps/ckpt_ema_0100000.pt", map_location="cpu")
model = UNet32(num_classes=10, cond_null_id=10, predict_sigma=True)
model.load_state_dict(ckpt["model"]); model.eval()

# 2) build diffusion (use the same T as training)
T = int(json.loads(ckpt["config"])["timesteps"])
diff = GaussianDiffusion(model, DiffusionConfig(image_size=32, timesteps=T, learn_sigma=True))

# 3) get a small real batch in [-1,1]
tfm = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize([0.5]*3,[0.5]*3)])
ds = datasets.ImageFolder("data/cifar10/train", transform=tfm)
x, y = torch.stack([ds[i][0] for i in range(32)]), torch.tensor([ds[i][1] for i in range(32)], dtype=torch.long)

# 4) pick a mid t and test reconstruction quality
t = torch.full((x.size(0),), T//2, dtype=torch.long)
noise = torch.randn_like(x)
xt = diff.q_sample(x, t, noise)
with torch.no_grad():
    eps = model(xt, t, y)[:, :3]
x0_pred = (xt - diff.sqrt_one_minus_alphas_cumprod[t][:,None,None,None]*eps) / diff.sqrt_alphas_cumprod[t][:,None,None,None]
mse = torch.mean((x0_pred.clamp(-1,1)-x).pow(2)).item()
print("mid-t MSE:", mse) # should be very low, e.g. < 0.02
