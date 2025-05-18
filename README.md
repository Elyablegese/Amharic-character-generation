# AmharicGPT(Optimized for AWS EC2 p3.2xlarge)

A character-level generative transformer model trained on Amharic text using the GPT architecture.


```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess data
python src/preprocess.py

# Train model
python src/train.py

# Generate text
python src/generate.py
