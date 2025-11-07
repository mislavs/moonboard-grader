# MoonBoard Generator

A Variational Autoencoder (VAE) based generator for creating synthetic MoonBoard climbing problems.

## Overview

This project uses a conditional VAE to generate novel climbing problems at specified difficulty grades. The VAE learns latent representations of climbing problems from the MoonBoard dataset and can generate new problems by sampling from this learned distribution.

## Installation

1. Install the shared moonboard_core package:
```bash
pip install -e ../moonboard_core
```

2. Install generator dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a new VAE model:
```bash
python main.py train --config config.yaml
```

### Generating Problems

Generate new climbing problems:
```bash
python main.py generate --checkpoint models/best_vae.pth --grade 6B+ --num-samples 10
```

## Data Format

The generator uses the shared dataset located at `../data/problems.json`. Problems are represented as 3-channel grids (start holds, middle holds, end holds) and conditioned on difficulty grade during training and generation.

