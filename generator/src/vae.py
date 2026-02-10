"""
Conditional Variational Autoencoder for MoonBoard problem generation.

The VAE learns to encode climbing problems into a latent space and can generate
new problems conditioned on difficulty grade.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for generating MoonBoard climbing problems.

    Architecture:
    - Input: 3x18x11 grid (start/middle/end holds)
    - Conditioning: Grade label (embedded to vector)
    - Latent space: Gaussian distribution with learned mu and logvar
    - Output: 3x18x11 grid with probabilities for each hold

    Args:
        latent_dim: Dimension of the latent space (default: 128)
        num_grades: Number of unique grade labels (default: 17)
        grade_embedding_dim: Dimension of grade embedding vector (default: 32)
    """

    def __init__(self, latent_dim=128, num_grades=17, grade_embedding_dim=32):
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_grades = num_grades
        self.grade_embedding_dim = grade_embedding_dim

        # Grade embedding layer
        self.grade_embedding = nn.Embedding(num_grades, grade_embedding_dim)

        # Encoder: 3x18x11 -> latent_dim*2 (mu and logvar)
        self.encoder = nn.Sequential(
            # Input: 3x18x11
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32x18x11
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64x9x6
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128x5x3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 256x3x2
        )

        # Calculate flattened size after conv layers: 256 * 3 * 2 = 1536
        self.encoder_output_size = 256 * 3 * 2

        # Latent space layers: encoder features + grade embedding
        encoder_conditioned_size = self.encoder_output_size + grade_embedding_dim
        self.fc_mu = nn.Linear(encoder_conditioned_size, latent_dim)
        self.fc_logvar = nn.Linear(encoder_conditioned_size, latent_dim)

        # Decoder input: latent + grade embedding
        self.fc_decode = nn.Linear(
            latent_dim + grade_embedding_dim, self.encoder_output_size
        )

        # Decoder: latent_dim + grade_embedding_dim -> 3x18x11
        self.decoder = nn.Sequential(
            # 256x3x2
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128x6x4
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=0
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64x11x7
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32x22x14
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            # 3x22x14
        )

        # Final layer to adjust to exact output size
        self.output_adjust = nn.Conv2d(3, 3, kernel_size=1)

    def encode(self, x, grade_labels):
        """
        Encode input grid to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, 3, 18, 11)
            grade_labels: Grade labels of shape (batch_size,)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        grade_emb = self.grade_embedding(grade_labels)
        h_cond = torch.cat([h, grade_emb], dim=1)
        mu = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, grade_labels):
        """
        Decode latent vector to grid, conditioned on grade.

        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            grade_labels: Grade labels of shape (batch_size,)

        Returns:
            x_recon: Reconstructed grid of shape (batch_size, 3, 18, 11)
        """
        # Embed grade
        grade_emb = self.grade_embedding(grade_labels)

        # Concatenate latent vector with grade embedding
        z_cond = torch.cat([z, grade_emb], dim=1)

        # Decode
        h = self.fc_decode(z_cond)
        h = h.view(h.size(0), 256, 3, 2)  # Reshape to 256x3x2
        x_recon = self.decoder(h)

        # Adjust to exact output size (3x18x11)
        x_recon = F.interpolate(
            x_recon, size=(18, 11), mode="bilinear", align_corners=False
        )
        x_recon = self.output_adjust(x_recon)

        return x_recon

    def forward(self, x, grade_labels):
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor of shape (batch_size, 3, 18, 11)
            grade_labels: Grade labels of shape (batch_size,)

        Returns:
            x_recon: Reconstructed grid
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x, grade_labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, grade_labels)
        return x_recon, mu, logvar

    def sample(self, num_samples, grade_labels, device="cpu"):
        """
        Generate new samples from the learned distribution.

        Args:
            num_samples: Number of samples to generate
            grade_labels: Grade labels for conditioning (shape: num_samples,)
            device: Device to generate samples on

        Returns:
            samples: Generated grids of shape (num_samples, 3, 18, 11)
        """
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z, grade_labels)
        return samples


def vae_loss(x_recon, x, mu, logvar, kl_weight=1.0):
    """
    Compute VAE loss: reconstruction loss + KL divergence loss.

    Args:
        x_recon: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        kl_weight: Weight for KL divergence term

    Returns:
        total_loss: Total VAE loss
        recon_loss: Reconstruction loss (BCE)
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction="sum")

    # KL divergence loss
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss
