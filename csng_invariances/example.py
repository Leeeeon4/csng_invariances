"""Module with an example gan.

THe model runs quickly to test certain things, e.g. wandb. The example is based
on: https://towardsdatascience.com/pytorch-and-gans-a-micro-tutorial-804855817a6b
"""

import torch
from models.discriminator import ExampleDiscriminator as Discriminator
from models.generator import ExampleGenerator as Generator
from models.gan import ExampleGAN as VanillaGAN
import wandb
from rich import print


def main():
    from time import time

    # initialize wandb
    wandb.init(project="_TestProject", entity="csng-cuni")

    # initialize model
    epochs = 150
    batches = 10
    generator = Generator(1)
    discriminator = Discriminator(1, [64, 32, 1])
    noise_fn = lambda x: torch.rand((x, 1), device="cpu")
    data_fn = lambda x: torch.randn((x, 1), device="cpu")
    gan = VanillaGAN(generator, discriminator, noise_fn, data_fn, device="cpu")
    loss_g, loss_d_real, loss_d_fake = [], [], []
    start = time()

    # wandb store config
    config = wandb.config
    config.learning_rate_discriminator = gan.lr_d
    config.architecture_discriminator = (
        "Fully connected w/leaky ReLU and last activation sigmoid"
    )
    config.learning_rate_generator = gan.lr_g
    config.architecture_generator = "Linear, leaky ReLU, linear, leaky ReLU, linear"
    config.betas = gan.betas
    config.epochs = epochs
    config.batches = batches
    config.batch_size = gan.batch_size

    for epoch in range(epochs):
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        for batch in range(batches):
            lg_, (ldr_, ldf_) = gan.train_step()
            loss_g_running += lg_
            loss_d_real_running += ldr_
            loss_d_fake_running += ldf_
        loss_g.append(loss_g_running / batches)
        loss_d_real.append(loss_d_real_running / batches)
        loss_d_fake.append(loss_d_fake_running / batches)
        wandb.log(
            {
                "loss_gen": loss_g[-1],
                "loss_discr_real": loss_d_real[-1],
                "loss_discr_fake": loss_d_fake[-1],
            }
        )
        print(
            f"Epoch {epoch+1}/{epochs} ({int(time() - start)}s):"
            f" G={loss_g[-1]:.3f},"
            f" Dr={loss_d_real[-1]:.3f},"
            f" Df={loss_d_fake[-1]:.3f}"
        )
    wandb.finish()


if __name__ == "__main__":
    main()
