# WoW Icons Diffusion Model Study

![image](https://github.com/user-attachments/assets/1722cf1e-6581-4b8a-a2fe-7de6aa2afdd9)

This repository contains my personal implementation of a diffusion model for generating World of Warcraft style icons. This project is a learning exercise, directly based on ["A Diffusion Model from Scratch in PyTorch"](https://github.com/acids-ircam/diffusion_models) and additional resources cited below.

## Project Context
![image](https://github.com/user-attachments/assets/7aa0660b-fc4a-4871-844d-8b416b719792)

- **Educational Purpose**: This implementation was created as a self-study exercise to understand diffusion models through hands-on practice
- **Hardware Limitations**: Due to computational constraints, the model wasn't fully optimized or trained for extended periods
- **Source Material**: The code and approach directly adapt existing implementations with modifications for the WoW dataset

## Implementation Details

### Dataset
![image](https://github.com/user-attachments/assets/41694747-1aba-423d-80af-8b53e64a4b24)

- **Name**: CleanIcons-MechagnomeEdition (version 11.1.0.59347-V4-1)
- **Source**: [GitHub - AcidWeb/Clean-Icons-Mechagnome-Edition](https://github.com/AcidWeb/Clean-Icons-Mechagnome-Edition)
- **Format**: TGA (Truevision Graphics Adapter) files with transparency support

### Model Architecture
- **Backbone**: Simplified U-Net architecture
- **Adaptations**: 
  - Modified for TGA format support
  - Optimized for Apple Silicon (MPS) compatibility
  - Timestep embedding with sinusoidal position encodings

### Technical Components
- **Forward Diffusion**: Implemented with linear beta schedule
- **Reverse Diffusion**: DDPM (Denoising Diffusion Probabilistic Model) approach
- **Loss Function**: L1 loss between predicted and actual noise
- **Data Pipeline**: Custom loader for WoW icon processing

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
2. Place WoW icons in the `./data/ICONS` directory
3. Run the notebook cells sequentially to understand each component

## Potential Extensions

### Text-Conditional Generation
To add text prompts (e.g., "create a frost spell icon"):

- **Required Changes**:
  - Implement a conditional U-Net with text encoder (CLIP/T5/BERT)
  - Add cross-attention layers to process text embeddings
  - Create or source text-icon paired training data

### Alternative Approaches
- **Score-Based Diffusion**: Implement Score-SDE for improved sample quality and flexibility
- **Architecture Improvements**: Add attention mechanisms, transformer blocks
- **Sampling Efficiency**: Implement DDIM or DPM-Solver for faster generation
- **Training Enhancements**: Dynamic thresholding, classifier guidance

### Domain-Specific Features
- **Style Control**: Conditioning on specific WoW expansion art styles
- **Class-Specific Generation**: Target spell vs. item vs. ability icons
- **Color Palette Conditioning**: Control color themes and motifs

## Requirements
- PyTorch 2.6.0+ with MPS support
- PIL (Python Imaging Library)
- numpy
- matplotlib
