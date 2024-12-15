# Text2Bricks : Transform Text to Buildable Lego Set

## Overview
The Text2Bricks project aims to create buildable LEGO sets in the 3D LEGO representation format (LDRAW) based on natural language input. This involves using a diffusion model to generate a 3D shape from the input and reinforcement learning (RL) to construct the LEGO set based on the generated shape.

### Final Projected Pipeline
- **Natural Language Input**: User provides a description of the desired LEGO model.
- **Diffusion Model**: Converts the input into a 3D shape.
- **Text2Brick Reinforcement Learning Model**: Builds the LEGO model by:
  - Utilizing a gym environment.
  - Leveraging a combination of CNN and GNN for processing.
  - Representing the LEGO world as a graph.
- **Output**: A buildable LEGO set in LDRAW format.

**Pipeline:**  
Natural Language Input → Diffusion Model (3D Shape) → Text2Brick RL Model (Gym + CNN + GNN) → Buildable LEGO Set (LDRAW Format)

---

## Step 1: Initial Proof of Concept
### Objective
Rebuild MNIST digits in LEGO LDRAW format. This simplified approach focuses on 2D reconstruction (ignoring the z-dimension) to reduce complexity in the initial stages of the project.

![POC Target Example](images/Reconstruction_Example.png)

### Reinforcement Learning Model Pipeline
#### Observations
1. **Target Image**: MNIST digit to rebuild.
2. **Current Build LEGO Shape**: Converted to grayscale image at each epoch.
3. **Reward Function**:
   - Reward = α * *brick_validity* + β * *IoU*
     - **IoU**: Intersection-over-Union between the target image and the current LEGO shape.
     - **brick_validity**: Boolean indicating whether the brick placement is legal (e.g., no flying bricks).

#### RL Model Architecture
- **Model Type**: TBD (Possibly Q-Learning).
- **Components**:
  - **CNN**: Processes the target and current build images (using a backbone from a pretrained model).
  - **GNN**: Processes the graph representation of the LEGO world.
  - **Fusion and Attention Layer**: TBD – should we include this?
  - **Output**: Predicts the next LEGO node in the graph (Brick Class) or its x, y coordinates (to be determined).

---

## LDRAW Format
LEGO sets are generated in LDRAW format. For details, see the official specification:  
[LDRAW File Format Documentation](https://www.ldraw.org/article/218.html)

---

## References
1. **Brick by Brick**:  
   [NeurIPS 2021 Paper](https://proceedings.neurips.cc/paper/2021/file/2d4027d6df9c0256b8d4474ce88f8c88-Paper.pdf)

2. **Learning to Build by Building Your Own Instructions**:  
   [arXiv 2410.01111](https://www.arxiv.org/pdf/2410.01111)


