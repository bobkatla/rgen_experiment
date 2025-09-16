# run once on the CFG checkpoint
import torch, json, os
from torchvision import datasets, transforms
from rgen.core.model import UNet32
from rgen.core.diffusion import GaussianDiffusion, DiffusionConfig

ckptp = "output/cifar10_unet_eps_cfg/ckpt_ema_0200000.pt"
ckpt = torch.load(ckptp, map_location="cpu")
T = int(json.loads(ckpt["config"])["timesteps"])
m = UNet32(num_classes=10, cond_null_id=10, predict_sigma=True); m.load_state_dict(ckpt["model"]); m.eval()
diff = GaussianDiffusion(m, DiffusionConfig(image_size=32, timesteps=T, learn_sigma=True))

tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
ds = datasets.ImageFolder("data/cifar10/train", transform=tfm)
x = torch.stack([ds[i][0] for i in range(64)]); y = torch.tensor([ds[i][1] for i in range(64)])

t = torch.full((x.size(0),), T//2, dtype=torch.long)
noise = torch.randn_like(x); xt = diff.q_sample(x, t, noise)

with torch.no_grad():
    eps_c = m(xt, t, y)[:, :3]
    y_null = torch.full_like(y, 10)
    eps_u = m(xt, t, y_null)[:, :3]

xa = diff.sqrt_alphas_cumprod[t][:,None,None,None]
xb = diff.sqrt_one_minus_alphas_cumprod[t][:,None,None,None]
x0_c = (xt - xb*eps_c)/xa
x0_u = (xt - xb*eps_u)/xa

print("mid-t MSE (cond):", torch.mean((x0_c.clamp(-1,1)-x).pow(2)).item())
print("mid-t MSE (uncond):", torch.mean((x0_u.clamp(-1,1)-x).pow(2)).item())

