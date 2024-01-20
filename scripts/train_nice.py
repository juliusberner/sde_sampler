"""
Adapted from https://github.com/fmu2/NICE
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision

from sde_sampler.distr import nice
from sde_sampler.distr.base import DATA_DIR

MNIST_SIZE = 28


def dequantize(x):
    """Dequantize data.

    Add noise sampled from Uniform(0, 1) to each pixel (in [0, 255]).

    Args:
        x: input tensor.
    Returns:
        dequantized data.
    """
    noise = torch.rand_like(x)
    return (x * 255.0 + noise) / 256.0


def prepare_data(x, mean=None, reverse=False):
    """Prepares data for NICE.

    In training mode, flatten and dequantize the input.
    In inference mode, reshape tensor into image size.

    Args:
        x: input minibatch.
        mean: center of original dataset.
        reverse: True if in inference mode, False if in training mode.
    Returns:
        transformed data.
    """
    if reverse:
        width = int(np.sqrt(x.shape[-1]))
        assert width * width == x.shape[-1]
        x += mean
        x = x.reshape((x.shape[0], 1, width, width))
    else:
        assert x.shape[-1] == x.shape[-2]
        x = dequantize(x)
        x = x.reshape(x.shape[0], -1)
        x -= mean
    return x


def train_nice(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model hyperparameters
    resize = args.resize
    fraction = resize / MNIST_SIZE
    batch_size = args.batch_size
    latent = args.latent
    max_iter = args.max_iter
    sample_size = args.sample_size
    coupling = 4
    mask_config = 1.0

    # optimization hyperparameters
    lr = args.lr
    min_lr = args.min_lr
    momentum = args.momentum
    decay = args.decay

    # shapes
    shape = (resize, resize)
    full_dim, mid_dim, hidden = (1 * resize * resize, int(1000 * fraction), 5)

    # directory
    filename = (
        "bs%d_" % batch_size
        + "%s_" % latent
        + "cp%d_" % coupling
        + "md%d_" % mid_dim
        + "hd%d_" % hidden
    )
    log_dir = Path(__file__).parents[1] / "logs" / "nice" / filename
    log_dir.mkdir(exist_ok=True, parents=True)

    mean = torch.load(DATA_DIR / "mnist_mean.pt").reshape((1, MNIST_SIZE, MNIST_SIZE))
    mean = torchvision.transforms.Resize(size=shape, antialias=True)(mean).reshape(
        (1, full_dim)
    )
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size=shape, antialias=True),
        ]
    )
    trainset = torchvision.datasets.MNIST(
        root=log_dir / "MNIST", train=True, download=True, transform=transforms
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    if latent == "normal":
        prior = torch.distributions.Normal(
            torch.tensor(0.0).to(device), torch.tensor(1.0).to(device)
        )
    elif latent == "logistic":
        prior = nice.StandardLogistic()

    flow = nice.NiceModel(
        prior=prior,
        coupling=coupling,
        in_out_dim=full_dim,
        mid_dim=mid_dim,
        hidden=hidden,
        mask_config=mask_config,
    ).to(device)
    optimizer = torch.optim.Adam(
        flow.parameters(), lr=lr, betas=(momentum, decay), eps=1e-4
    )

    gamma = (min_lr / lr) ** (1 / max_iter)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    total_iter = 0
    train = True
    running_loss = 0

    while train:
        for _, data in enumerate(trainloader, 1):
            flow.train()  # set to training mode
            if total_iter == max_iter:
                train = False
                break

            total_iter += 1
            optimizer.zero_grad()  # clear gradient tensors

            inputs, _ = data
            inputs = prepare_data(inputs, mean=mean).to(device)

            # log-likelihood of input minibatch
            loss = -flow(inputs).mean()
            running_loss += float(loss)

            # backprop and update parameters
            loss.backward()
            optimizer.step()
            scheduler.step()

            if total_iter % 1000 == 0:
                mean_loss = running_loss / 1000
                bit_per_dim = (mean_loss + np.log(256.0) * full_dim) / (
                    full_dim * np.log(2.0)
                )
                print(
                    "iter %s:" % total_iter,
                    "loss = %.3f" % mean_loss,
                    "bits/dim = %.3f" % bit_per_dim,
                    "lr = %.5f" % optimizer.param_groups[0]["lr"],
                )
                running_loss = 0.0

                flow.eval()  # set to inference mode
                with torch.no_grad():
                    z, _ = flow.f(inputs)
                    reconst = flow.g(z).cpu()
                    reconst = prepare_data(reconst, mean=mean, reverse=True)
                    samples = flow.sample(sample_size).cpu()
                    samples = prepare_data(samples, mean=mean, reverse=True)
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(reconst),
                        log_dir / f"reconstruction_iter{total_iter}.png",
                    )
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(samples),
                        log_dir / f"samples_iter{total_iter}.png",
                    )

    print("Finished training!")

    torch.save(
        {
            "total_iter": total_iter,
            "model_state_dict": flow.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "batch_size": batch_size,
            "latent": latent,
            "coupling": coupling,
            "mid_dim": mid_dim,
            "hidden": hidden,
            "mask_config": mask_config,
        },
        DATA_DIR / "nice.pt",
    )

    print("Checkpoint Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MNIST NICE PyTorch implementation")
    parser.add_argument("--resize", help="resize width/height.", type=int, default=14)
    parser.add_argument(
        "--batch_size", help="number of images in a mini-batch.", type=int, default=200
    )
    parser.add_argument(
        "--latent", help="latent distribution.", type=str, default="logistic"
    )
    parser.add_argument(
        "--max_iter", help="maximum number of iterations.", type=int, default=10000
    )
    parser.add_argument(
        "--sample_size", help="number of images to generate.", type=int, default=64
    )
    parser.add_argument("--lr", help="initial learning rate.", type=float, default=1e-3)
    parser.add_argument(
        "--min_lr", help="minimal learning rate.", type=float, default=1e-4
    )
    parser.add_argument(
        "--momentum", help="beta1 in Adam optimizer.", type=float, default=0.9
    )
    parser.add_argument(
        "--decay", help="beta2 in Adam optimizer.", type=float, default=0.999
    )
    args = parser.parse_args()
    train_nice(args)
