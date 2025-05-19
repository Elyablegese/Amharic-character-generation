# AmharicGPT (Optimized for AWS EC2 p3.2xlarge)

A character-level generative transformer model trained on Amharic text using the GPT architecture. Optimized for GPU acceleration on AWS EC2 p3.2xlarge instances with NVIDIA V100 GPUs.

## Sample Outputs

### Short Prompt Example
![Short Prompt Output](https://github.com/user-attachments/assets/2d10f305-34d9-4c43-ba63-343dfcf56019%20 )

### Long Prompt Example
![Long Prompt Output](https://github.com/user-attachments/assets/5bb5bf00-dcb5-44ba-b8a6-9f6535d29841%20 )

## Installation

```bash
# Clone repository
git clone https://github.com/Elyablegese/Amharic-character-generation.git   
cd Amharic-character-generation

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Preprocess data
python src/preprocess.py

# Train model (GPU recommended)
python src/train.py \
  --data-dir data/processed \
  --save-dir results/checkpoints \
  --epochs 50 \
  --batch-size 64

# Generate text from trained model
python src/generate.py

@mastersthesis{legesse2025chartransformer,
title = {A Character-Level Transformer for Amharic Text Generation},
author = {Legesse, Eliab},
school = {KOTEBE UNIVERSITY OF EDUCATION},
year = {2025}
}
