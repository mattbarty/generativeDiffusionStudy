# Diffusion Model Study
https://github.com/user-attachments/assets/fdbc5fb1-7833-4600-aab0-f7840acb4f72

This repository contains my personal implementation of a diffusion model for generating MNIST handwritten digits. This project is a learning exercise, directly based on ["A Diffusion Model from Scratch in PyTorch"](https://github.com/acids-ircam/diffusion_models) and additional resources cited below. This is also adapted for M1 pro (mps).

## Project Context
![image](https://github.com/user-attachments/assets/d9e88cd7-b19a-4328-a6e4-56055a09ac31)
- **Educational Purpose**: This implementation was created as a self-study exercise to understand diffusion models through hands-on practice
- **Hardware Limitations**: Due to computational constraints, the model wasn't fully optimized or trained for extended periods
- **Source Material**: The code and approach directly adapt existing implementations with modifications for the MNIST dataset

## Implementation Details
### Dataset
![image](https://github.com/user-attachments/assets/e24d0b63-9a36-4de3-8d8e-c19baa68de17)
- **Name**: MNIST Database of handwritten digits
- **Source**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- **Format**: 28×28 grayscale images of handwritten digits (0-9)

### Model Architecture
- **Backbone**: Simplified U-Net architecture
- **Adaptations**: 
  - Modified for grayscale image support
  - Optimized for Apple Silicon (MPS) compatibility
  - Timestep embedding with sinusoidal position encodings

### Technical Components
- **Forward Diffusion**: Implemented with linear beta schedule
- **Reverse Diffusion**: DDPM (Denoising Diffusion Probabilistic Model) approach
- **Loss Function**: L1 loss between predicted and actual noise
- **Data Pipeline**: Standard PyTorch MNIST data loader with normalization

## Primary References
This implementation directly adapts code and concepts from:
1. **Main Reference**: ["A Diffusion Model from Scratch in PyTorch"](https://github.com/acids-ircam/diffusion_models)
2. **DDPM Implementation**: [Denoising Diffusion PyTorch](https://github.com/lucidrains/denoising-diffusion-pytorch) by Phil Wang
3. **Tutorial**: [Huggingface Diffusion Models Tutorial](https://github.com/huggingface/diffusion-models-class) by Niels Rogge and Kashif Rasul

### Academic Papers
- Ho et al. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Nichol & Dhariwal (2021). [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- Dhariwal & Nichol (2021). [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)

## Usage
This notebook serves primarily as a learning resource:
1. Ensure PyTorch is installed with MPS support
2. The MNIST dataset will be automatically downloaded by torchvision
3. Run the notebook cells sequentially to understand each component

## Potential Extensions
### Conditional Generation
To add conditional digit generation (e.g., "generate digit 7"):
- **Required Changes**:
  - Implement a conditional U-Net with label embedding
  - Add cross-attention layers or class embeddings
  - Use the labels already provided in the MNIST dataset

### Alternative Approaches
- **Score-Based Diffusion**: Implement Score-SDE for improved sample quality and flexibility
- **Architecture Improvements**: Add attention mechanisms, transformer blocks
- **Sampling Efficiency**: Implement DDIM or DPM-Solver for faster generation
- **Training Enhancements**: Dynamic thresholding, classifier guidance

### Domain-Specific Features
- **Style Control**: Conditioning on specific handwriting styles
- **Multi-digit Generation**: Extending to generate multiple digits or simple equations
- **Handwriting Transfer**: Generate digits in the style of specific writers

## Requirements
- PyTorch 2.6.0+ with MPS support
- torchvision
- numpy
- matplotlib
