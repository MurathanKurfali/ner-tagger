
# ner-tagger
An implementation of a named entity recognition system through fine-tuning transformer-based models.

This project provides an easy-to-use framework for training and evaluating NER systems using the "Babelscape/multinerd" dataset. 


## Getting Started

To get started with this project, follow these steps:

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone [repository URL]
cd ner-tagger
```

### 2. Create and Activate a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv ner_tagger_env
```
Activate the virtual environment:
- On Windows:
  ```bash
  ner_tagger_env\Scripts\activate
  ```
- On MacOS/Linux:
  ```bash
  source ner_tagger_env/bin/activate
  ```

## Requirements

This project requires Python 3.6 or later, along with the several packages.
You can install all the needed packages using pip:

```bash
pip install -r requirements.txt
```

## Running

### 1) Preparing the Dataset

Prepare the target dataset for fine-tuning. The "prepare_dataset" script provides two configurations:

- **System A (All Tags)**: Retains all the tags in the target dataset.
- **System B (Subset of Tags)**: Retains only 5 specific tags, removing the others but keeping the associated words.

#### Sample Usage

To use the script, configure it via command-line arguments:

```bash
python prepare_dataset.py --dataset_name "Babelscape/multinerd" --language "en" --output_dir "data" --tag-set "A"
```

##### Arguments

- `--dataset_name`: Name of the dataset to load. Default is "Babelscape/multinerd".
- `--language`: Language to filter the dataset by. Default is "en" (English).
- `--output_dir`: Directory to save the output file. Default is "data".
- `--tag-set`: Specifies the tag set to use. Choose 'A' for all tags (System A) or 'B' for a subset of tags (System B). Default is 'A'.

Please note that the script is tailored to the multinerd dataset. However, it is straightforward to adapt to any NER dataset. Based on your dataset, you will need to change the label mappings in `config.py`.

### 2) Fine-tuning the Model

To fine-tune the model with the default parameters, simply call the `run.sh` script:

```bash
./run.sh <system-name> <target-model> <loss_choice>
```
This script automatically fine-tunes the target model four times  with different seeds for robust evaluation. Replace `<system-name>`, `<target-model>`, and `<loss_choice>` with appropriate values for your experiment.
```bash
e.g ./run.sh "a" "bert-base-cased" "False"
```

Trial scores

| Metric | ANIM | BIO | CEL | DIS | EVE | FOOD | INST | LOC | MEDIA | MYTH | ORG | PER | PLANT | TIME | VEHI | overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Precision | 0.724 | 0.420 | 0.757 | 0.743 | 0.952 | 0.657 | 0.688 | 0.995 | 0.954 | 0.771 | 0.977 | 0.993 | 0.631 | 0.826 | 0.829 | 0.939 |
| Recall | 0.781 | 0.812 | 0.829 | 0.795 | 0.976 | 0.693 | 0.792 | 0.994 | 0.974 | 0.828 | 0.984 | 0.995 | 0.738 | 0.839 | 0.906 | 0.955 |
| F1 | 0.751 | 0.553 | 0.791 | 0.768 | 0.964 | 0.674 | 0.732 | 0.995 | 0.964 | 0.798 | 0.981 | 0.994 | 0.680 | 0.833 | 0.866 | 0.947 |
Process finished with exit code 0
