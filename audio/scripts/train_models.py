# scripts/train_models.py
"""
Training script for audio synthesis models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import argparse
import yaml
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

from audio_synth.core.generators.gan import AudioGenerator, AudioDiscriminator
from audio_synth.core.generators.vae import AudioVAE
from audio_synth.core.utils.config import AudioConfig, GenerationConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    """Dataset for loading audio files"""
    
    def __init__(self, 
                 data_dir: str, 
                 sample_rate: int = 22050,
                 duration: float = 5.0,
                 augment: bool = False):
        
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.augment = augment
        self.target_length = int(sample_rate * duration)
        
        # Find all audio files
        self.audio_files = []
        for ext in ["*.wav", "*.mp3", "*.flac"]:
            self.audio_files.extend(list(self.data_dir.glob(f"**/{ext}")))
        
        logger.info(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load audio
        audio, orig_sr = torchaudio.load(str(audio_path))
        
        # Resample if necessary
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Adjust length
        if audio.shape[1] > self.target_length:
            # Random crop
            start = torch.randint(0, audio.shape[1] - self.target_length + 1, (1,))
            audio = audio[:, start:start + self.target_length]
        elif audio.shape[1] < self.target_length:
            # Pad with zeros
            padding = self.target_length - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        # Apply augmentation
        if self.augment:
            audio = self._augment_audio(audio)
        
        # Normalize
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        return audio.squeeze()
    
    def _augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation"""
        
        # Random volume scaling
        if torch.rand(1) < 0.5:
            volume_scale = 0.5 + torch.rand(1) * 0.5  # 0.5 to 1.0
            audio = audio * volume_scale
        
        # Add noise
        if torch.rand(1) < 0.3:
            noise_level = torch.rand(1) * 0.05  # Up to 5% noise
            noise = torch.randn_like(audio) * noise_level
            audio = audio + noise
        
        # Time shifting
        if torch.rand(1) < 0.3:
            shift_samples = int(torch.randint(-1000, 1001, (1,)))
            if shift_samples > 0:
                audio = torch.cat([torch.zeros(1, shift_samples), audio[:, :-shift_samples]], dim=1)
            elif shift_samples < 0:
                audio = torch.cat([audio[:, -shift_samples:], torch.zeros(1, -shift_samples)], dim=1)
        
        return audio

class GANTrainer:
    """Trainer for GAN-based audio generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.generator = AudioGenerator(
            latent_dim=config["model"]["latent_dim"],
            output_length=config["audio"]["target_length"],
            num_channels=1
        ).to(self.device)
        
        self.discriminator = AudioDiscriminator(
            input_length=config["audio"]["target_length"],
            num_channels=1
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config["training"]["lr_g"],
            betas=(config["training"]["beta1"], config["training"]["beta2"])
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config["training"]["lr_d"],
            betas=(config["training"]["beta1"], config["training"]["beta2"])
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Tracking
        self.g_losses = []
        self.d_losses = []
        
    def train(self, dataloader: DataLoader, num_epochs: int):
        """Train the GAN"""
        
        logger.info(f"Starting GAN training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, real_audio in enumerate(progress_bar):
                batch_size = real_audio.size(0)
                real_audio = real_audio.unsqueeze(1).to(self.device)
                
                # Labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                # Real samples
                real_output = self.discriminator(real_audio)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # Fake samples
                noise = torch.randn(batch_size, self.config["model"]["latent_dim"]).to(self.device)
                fake_audio = self.generator(noise)
                fake_output = self.discriminator(fake_audio.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                fake_output = self.discriminator(fake_audio)
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
                
                # Update tracking
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "G_Loss": f"{g_loss.item():.4f}",
                    "D_Loss": f"{d_loss.item():.4f}"
                })
                
                # Log to wandb
                if batch_idx % 50 == 0:
                    wandb.log({
                        "g_loss": g_loss.item(),
                        "d_loss": d_loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx
                    })
            
            # Epoch statistics
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            
            logger.info(f"Epoch {epoch+1} - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
            
            # Generate samples
            if (epoch + 1) % 5 == 0:
                self.generate_samples(epoch + 1)
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        
        checkpoint = {
            "epoch": epoch,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "g_losses": self.g_losses,
            "d_losses": self.d_losses,
            "config": self.config
        }
        
        checkpoint_path = Path(self.config["training"]["checkpoint_dir"])
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path / f"gan_checkpoint_epoch_{epoch}.pt")
        logger.info(f"Checkpoint saved for epoch {epoch}")
    
    def generate_samples(self, epoch: int, num_samples: int = 5):
        """Generate sample audio"""
        
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.config["model"]["latent_dim"]).to(self.device)
            fake_audio = self.generator(noise)
        
        # Save samples
        samples_dir = Path(self.config["training"]["samples_dir"])
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        for i, audio in enumerate(fake_audio):
            filename = samples_dir / f"epoch_{epoch}_sample_{i+1}.wav"
            torchaudio.save(str(filename), audio.cpu(), self.config["audio"]["sample_rate"])
        
        self.generator.train()
        logger.info(f"Generated {num_samples} samples for epoch {epoch}")

class VAETrainer:
    """Trainer for VAE-based audio generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.vae = AudioVAE(
            input_length=config["audio"]["target_length"],
            latent_dim=config["model"]["latent_dim"],
            num_channels=1
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.vae.parameters(),
            lr=config["training"]["lr"]
        )
        
        # Beta for beta-VAE
        self.beta = config["model"].get("beta", 1.0)
        
        # Tracking
        self.losses = []
        self.recon_losses = []
        self.kl_losses = []
    
    def vae_loss(self, recon_x, x, mu, logvar):
        """VAE loss function"""
        
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Combined loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train(self, dataloader: DataLoader, num_epochs: int):
        """Train the VAE"""
        
        logger.info(f"Starting VAE training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Beta: {self.beta}")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, audio in enumerate(progress_bar):
                audio = audio.unsqueeze(1).to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                recon_audio, mu, logvar = self.vae(audio)
                
                # Calculate loss
                loss, recon_loss, kl_loss = self.vae_loss(recon_audio, audio, mu, logvar)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update tracking
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Recon": f"{recon_loss.item():.4f}",
                    "KL": f"{kl_loss.item():.4f}"
                })
                
                # Log to wandb
                if batch_idx % 50 == 0:
                    wandb.log({
                        "total_loss": loss.item(),
                        "recon_loss": recon_loss.item(),
                        "kl_loss": kl_loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx
                    })
            
            # Epoch statistics
            avg_loss = epoch_loss / len(dataloader)
            avg_recon_loss = epoch_recon_loss / len(dataloader)
            avg_kl_loss = epoch_kl_loss / len(dataloader)
            
            self.losses.append(avg_loss)
            self.recon_losses.append(avg_recon_loss)
            self.kl_losses.append(avg_kl_loss)
            
            logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
            
            # Generate samples
            if (epoch + 1) % 5 == 0:
                self.generate_samples(epoch + 1)
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        
        checkpoint = {
            "epoch": epoch,
            "model": self.vae.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "losses": self.losses,
            "recon_losses": self.recon_losses,
            "kl_losses": self.kl_losses,
            "config": self.config
        }
        
        checkpoint_path = Path(self.config["training"]["checkpoint_dir"])
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path / f"vae_checkpoint_epoch_{epoch}.pt")
        logger.info(f"Checkpoint saved for epoch {epoch}")
    
    def generate_samples(self, epoch: int, num_samples: int = 5):
        """Generate sample audio"""
        
        self.vae.eval()
        
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(num_samples, self.config["model"]["latent_dim"]).to(self.device)
            generated_audio = self.vae.decode(z)
        
        # Save samples
        samples_dir = Path(self.config["training"]["samples_dir"])
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        for i, audio in enumerate(generated_audio):
            filename = samples_dir / f"epoch_{epoch}_sample_{i+1}.wav"
            torchaudio.save(str(filename), audio.cpu(), self.config["audio"]["sample_rate"])
        
        self.vae.train()
        logger.info(f"Generated {num_samples} samples for epoch {epoch}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    """Main training script"""
    
    parser = argparse.ArgumentParser(description="Train Audio Synthesis Models")
    parser.add_argument("--config", type=str, required=True, help="Training configuration file")
    parser.add_argument("--model", type=str, choices=["gan", "vae"], required=True, help="Model type to train")
    parser.add_argument("--data-dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project="audio-synthesis",
            config=config,
            name=f"{args.model}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create dataset
    dataset = AudioDataset(
        data_dir=args.data_dir,
        sample_rate=config["audio"]["sample_rate"],
        duration=config["audio"]["duration"],
        augment=config["training"].get("augment", False)
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize trainer
    if args.model == "gan":
        trainer = GANTrainer(config)
    elif args.model == "vae":
        trainer = VAETrainer(config)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # TODO: Implement checkpoint loading
    
    # Start training
    trainer.train(dataloader, config["training"]["epochs"])
    
    # Save final model
    final_path = Path(config["training"]["checkpoint_dir"]) / f"{args.model}_final.pt"
    if args.model == "gan":
        torch.save({
            "generator": trainer.generator.state_dict(),
            "discriminator": trainer.discriminator.state_dict(),
            "config": config
        }, final_path)
    else:  # VAE
        torch.save({
            "model": trainer.vae.state_dict(),
            "config": config
        }, final_path)
    
    logger.info(f"Training completed! Final model saved to: {final_path}")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()

# ============================================================================
