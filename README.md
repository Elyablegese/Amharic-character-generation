# AmharicGPT(Optimized for AWS EC2 p3.2xlarge)

A character-level generative transformer model trained on Amharic text using the GPT architecture.


```bash
Installation

Clone the repository:

git clone https://github.com/Elyablegese/Amharic-character-generation.git
cd Amharic-character-generation

# Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
pip install -r requirements.txt

# Preprocess data
python src/preprocess.py

# Train model
python src/train.py \
  --data-dir data/processed \
  --save-dir results/checkpoints \
  --epochs 50 \
  --batch-size 64
# Generate text
python src/generate.py

@mastersthesis{legesse2025chartransformer,
  title = {A Character-Level Transformer for Amharic Text Generation},
  author = {Legesse, Eliab},
  school = {Your University},
  year = {2025}
}

# sample output
Short Prompt
![image17](https://github.com/user-attachments/assets/2d10f305-34d9-4c43-ba63-343dfcf56019)
Long Prompt
![image18](https://github.com/user-attachments/assets/5bb5bf00-dcb5-44ba-b8a6-9f6535d29841)
