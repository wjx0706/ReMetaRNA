# ReMetaRNA
A MPNN-based RNA inverse folding model developed for the 3rd World Science Intelligence Competition organized by SAIS.

## Project Structure

```text
├─ReMetaRNA
│  ├─data.py            # Defines dataset construction
│  ├─modeling_rna.py    # Defines the model, including a contrastive learning model and an inference-specific model
│  ├─preprocessing.py   # Preprocessing used for training the model
│  ├─trainer.py         # Defines the training loop as well as evaluation and testing logic
│  ├─train.py           # Script to run model training
│  ├─main.py            # Script to run evaluation, including preprocessing logic. Note: only runs inference!
│  ├─requirements.txt   # Required Python packages for the model
│  └─final_model.pt     # Trained model weights
```


## Environment Requirements

### Operating System

Linux version 5.15.0-126-generic (buildd@lcy02-amd64-101)
(gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0, GNU ld (GNU Binutils for Ubuntu) 2.38)

### Python Environment

- Python 3.10.13

### Required Python Packages

```text
biopython==1.85
torch-geometric==2.6.1
pandas==2.2.3
tqdm==4.65.0
matplotlib==3.10.3
torch==2.7.0+cu126
```

### CUDA Dependencies

- CUDA: 12.6
- cuDNN: 9.5.1
