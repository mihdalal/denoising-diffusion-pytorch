from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l2',            # L1 or L2
    beta_schedule='linear',           # linear or cosine
)

trainer = Trainer(
    diffusion,
    '../sp23_vlr_hw2/datasets/CUB_200_2011_32/',
    train_batch_size = 256,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
)
torch.backends.cudnn.benchmark = True # speed up training
trainer.train()
