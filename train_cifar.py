import torch
import torchvision
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from cleanfid import fid as cleanfid

def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")

@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.sample_given_z(z, batch_size)*255)
    score = cleanfid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="train",
    )
    return score

torch.backends.cudnn.benchmark = True # speed up training
# load dataset from the hub
image_size = 32
channels = 3
batch_size = 128

# define image transformations (e.g. using torchvision)
transform = Compose([
                ToTensor(),
                Lambda(lambda t: (t * 2) - 1)
            ])

dataset = torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = channels,
    self_condition=True
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = 250,   # number of steps
    loss_type = 'l1',    # L1 or L2
).cuda()
scaler = torch.cuda.amp.GradScaler()
amp_enabled = False
num_iterations = 10**6
iters = 0
pbar = tqdm(total = num_iterations)
fids_list = []
iters_list = []
prefix = "data_diffusion/"
os.makedirs(prefix, exist_ok=True)
while iters < num_iterations:
    for training_images, _ in dataloader:
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            training_images = training_images.cuda() # images are normalized from 0 to 1
            loss = diffusion(training_images)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.update(1)
        pbar.set_description("Iteration: {}, Loss: {}".format(iters, loss))
        iters += 1
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            if iters % 1000 == 0:
                fid = get_fid(diffusion, "cifar10", 32, 32*32*3, batch_size=256, num_gen=1024)
                print(f"Iteration {iters} FID: {fid}")
                fids_list.append(fid)
                iters_list.append(iters)
                sampled_images = diffusion.sample(batch_size = 100)
                sampled_images.shape # (4, 3, 128, 128)
                # save images to disk in grid form
                save_image(
                    sampled_images.data.float(),
                    prefix + "samples_{}.png".format(iters),
                    nrow=10,
                )
                save_plot(
                    iters_list,
                    fids_list,
                    xlabel="Iterations",
                    ylabel="FID",
                    title="FID vs Iterations",
                    filename=prefix + "fid_vs_iterations",
                )

score = get_fid(diffusion, "cifar10", 32, 32*32*3, batch_size=256, num_gen=50_000)
print("FID: ", score)
