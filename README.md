# Generative Pseudo Labeling (GPL)

## Usage
First install the repo:
```bash
pip install -e .
```
One needs then to download some BeIR dataset as the `evaluation_data`.
And then call the `gpl.train` to do the training and evaluation all in one from the CLI: (Run `python -m gpl.train` --help for the descriptions of the arguments)
```bash
python -m gpl.train \
    --generated_path $generated_path \
    --base_ckpt $base_ckpt \
    --output_dir $output_dir \
    --mnrl_output_dir $mnrl_output_dir \
    --evaluation_data $evaluation_data \
    --evaluation_output $evaluation_output \
    --mnrl_evaluation_output $mnrl_evaluation_output \
    --retrievers "msmarco-distilbert-base-v3" "msmarco-MiniLM-L-6-v3"
```

## Code Structure

```bash
.
├── gpl
│   ├── toolkit  # Code/Toolkit for the components
│   │   ├── __init__.py
│   │   ├── dataset.py  # For loading the generated data and sampling examples
│   │   ├── evaluation.py  # For evaluation
│   │   ├── loss.py  # Margin-MSE loss; pseudo labeling is applied on the fly
│   │   ├── mine.py  # Hard-negative mining
│   │   ├── mnrl.py  # The training objective for QGen
│   │   ├── qgen.py  # Query generation
│   │   └── resize.py  # For resizing the corpus if needed
│   └── train.py  # Training and evaluation. Entry point with `python -m gpl.train` after installation
├── README.md
└── setup.py
```