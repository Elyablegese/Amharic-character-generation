# AmharicGPT (Optimized for AWS EC2 p3.2xlarge)

A character-level generative transformer model trained on Amharic text using the GPT architecture. Optimized for GPU acceleration on AWS EC2 p3.2xlarge instances with NVIDIA V100 GPUs.

### Features

- Character-level text generation for the Amharic language.
- Trained on a cleaned and preprocessed Amharic text corpus.
- Utilizes modern deep learning techniques for sequence prediction.

### Dataset

The dataset consists of a large collection of 5GB of Amharic text. 

## Installation

```bash
# Clone repository
git clone https://github.com/Elyablegese/Amharic-character-generation.git   
cd Amharic-character-generation

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Preprocess data
python src/preprocess.py

```

### Training

Training the Amharic text generation model on a CPU is possible but highly inefficient due to the computational demands of transformer-based architectures. For practical purposes, GPU acceleration is strongly recommended. As a benchmark, training on an NVIDIA GeForce RTX 3040 Super with the default hyperparameters (50 epochs, batch size of 64, learning rate of 1e-4) completed in approximately around 4 hours.

i. Preprocess the dataset (preprocess.py): Cleans the Amharic dataset by removing noise and irrelevant content.

ii. The training script (train.py) includes the following steps:

    1. Model Building: Define and compile the model.
    2. Data Preprocessing: Load and clean the dataset, then convert it to sequences of characters.
    3. Training: Train the model using the training data, with validation on the validation set.
    4. Hyperparameters such as batch size, learning rate, and number of epochs can be adjusted in the script.
```    
# Train model (GPU recommended)
python src/train.py \
  --data-dir data/processed \
  --save-dir results/checkpoints \
  --epochs 50 \
  --batch-size 64

# Generate text from trained model
python src/generate.py
```
## Sample Outputs

### Short Prompt Example
![Short Prompt Output](https://github.com/user-attachments/assets/2d10f305-34d9-4c43-ba63-343dfcf56019%20 )

### Long Prompt Example
![Long Prompt Output](https://github.com/user-attachments/assets/5bb5bf00-dcb5-44ba-b8a6-9f6535d29841%20 )
```
@mastersthesis{legesse2025chartransformer,
title = {A Character-Level Transformer for Amharic Text Generation},
author = {Legesse, Eliab},
school = {KOTEBE UNIVERSITY OF EDUCATION},
year = {2025}
}
