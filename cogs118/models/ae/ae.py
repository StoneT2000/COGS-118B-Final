import torch
from torch import nn
from torch.nn import functional as F
from typing import List

class AE(nn.Module):


    def __init__(self,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(AE, self).__init__()


        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)


        # Build Decoder
        modules = []

        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(self.hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def encode(self, input) -> List:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        return result

    def decode(self, z):
        result = z.view(-1, self.hidden_dims[0], 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    def forward(self, input, **kwargs) -> List:
        x = self.encode(input)
        return self.decode(x)

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the AE loss function, which is simply a pixel wise loss
        """
        recons = args[0]
        input = args[1]
        recons_loss = F.mse_loss(recons, input)
        loss = recons_loss
        return {'loss': loss}

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]