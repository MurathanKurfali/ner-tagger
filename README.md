
#  NER Tagging with Transformer Models
An implementation of a named entity recognition system through fine-tuning transformer-based models.

This project provides an easy-to-use framework for training and evaluating NER systems using the "Babelscape/multinerd" dataset. 


## Getting Started

To get started with this project, follow these steps:

#### 1. Clone the Repository
Clone the repository to your local machine:

```bash
git clone https://github.com/MurathanKurfali/ner-tagger.git
cd ner-tagger
```

#### 2. Setting Up Your NER Tagging Environment

Create a Python virtual environment:

```bash
python -m venv ner_tagger_env
```
Activate the virtual environment:
  ```bash
  source ner_tagger_env/bin/activate
  ```

### Install the Requirements

This project requires Python 3.6 or later, along with the several packages.
You can install all the needed packages using pip:

```bash
pip install -r requirements.txt
```

## Guide: Dataset Preparation and Model Fine-Tuning

### 1) Preparing the Dataset

Prepare the target dataset for fine-tuning. The "prepare_dataset" script provides two configurations:

- **System A (All Tags)**: Retains all the tags in the target dataset.
- **System B (Subset of Tags)**: Retains only 5 specific tags [PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS),
ANIMAL(ANIM)]. Other tags are removed but the associated words are kept in order to keep the sentences intact.

#### Sample Usage

To use the script, configure it via command-line arguments:

```bash
python prepare_dataset.py --dataset_name "Babelscape/multinerd" --language "en" --output_dir "data" --tag-set "A"
```

##### Arguments

- `--dataset_name`: Name of the dataset to load. Default is "Babelscape/multinerd".
- `--language`: Language to filter the dataset by. Default is "en" (English).
- `--output_dir`: Directory to save the output file. Default is "data".
- `--tag-set`: Specifies the tag set to use. Choose 'A' (or 'a') for all tags (System A) or 'B' (or 'b') for a subset of tags (System B). Default is 'A'.

Please note that the script is tailored to the multinerd dataset. However, it is straightforward to adapt to any NER dataset. Based on your dataset, you will need to change the label mappings in `config.py`.

### 2) Fine-tuning the Model

To fine-tune the model with the default parameters, simply call the `run.sh` script:

```bash
./run.sh <system-name> <target-model> <loss_choice> [<data-dir>] [<number-of-runs>]
```
This script automatically fine-tunes the target model multiple times with different seeds for robust evaluation. The data directory is an optional fourth argument, with a default value of "data." (**Do not forget to set this variable if you use a non-default output_dir in the prepare_dataset script.**) Replace <system-name>, <target-model>, and <loss_choice> with appropriate values for your experiment. The number of runs can be optionally set by providing it as the fifth argument. If no fifth argument is provided, the script defaults to running 4 times.

```bash
e.g ./run.sh "a" "bert-base-cased" "False" 10
```

### 3) Evaluation
The performance of each fine-tuning run is saved in output_dir (e.g., see saved_models/system_a/<bert-name>_<seed>/predict_results.json). However, to calculate the overall performance across runs, you can simply run the evaluate_predictions.py script:
```bash
python scripts/evaluate_predictions.py "saved_models" <data_dir> <target-model> [<output-format>]
```
This script will calculate precision, recall, and F1-score for each tag and, by default, save the results as an HTML table. However, you have the option to save the results in CSV, HTML, or Markdown formats.


### Quick Start Example

The following example demonstrates the preparation of the dataset with the tag set "A," followed by running the main script with the default parameters.

```bash
python prepare_dataset.py --tag-set "a"
./run.sh "a" "bert-base-cased" "False"
python scripts/evaluate_predictions.py "saved_models" "data" "bert-base-cased" 

```
### 4) Results

Below are the results of the model on both configurations, using bert-base-cased and xlnet-base-cased models. The results are 
averaged over 4 runs.

### bert-base-cased
#### System A
|           |        ANIM |       BIO |       CEL |         DIS |        EVE |        FOOD |      INST |          LOC |      MEDIA |      MYTH |         ORG |          PER |       PLANT |       TIME |      VEHI |    overall |
|:----------|------------:|----------:|----------:|------------:|-----------:|------------:|----------:|-------------:|-----------:|----------:|------------:|-------------:|------------:|-----------:|----------:|-----------:|
| precision |    0.730958 |  0.42372  |  0.748029 |    0.744328 |   0.954162 |    0.653681 |  0.688988 |     0.994802 |   0.962849 |  0.796495 |    0.979856 |     0.993224 |    0.635877 |   0.826381 |  0.862262 |   0.941031 |
| recall    |    0.778367 |  0.75     |  0.829268 |    0.796443 |   0.975142 |    0.689488 |  0.770833 |     0.994677 |   0.972162 |  0.84375  |    0.984436 |     0.995204 |    0.730425 |   0.846886 |  0.921875 |   0.955059 |
| f1        |    0.753792 |  0.538377 |  0.786287 |    0.769469 |   0.964524 |    0.671099 |  0.725046 |     0.994739 |   0.967441 |  0.81867  |    0.982138 |     0.994213 |    0.679704 |   0.836433 |  0.890858 |   0.94799  |
| number    | 3208        | 16        | 82        | 1518        | 704        | 1132        | 24        | 24048        | 916        | 64        | 6618        | 10530        | 1788        | 578        | 64        | nan        |
#### System B
|           |        ANIM |         DIS |          LOC |         ORG |          PER |    overall |
|:----------|------------:|------------:|-------------:|------------:|-------------:|-----------:|
| precision |    0.731486 |    0.751412 |     0.994121 |    0.98054  |     0.993268 |   0.964451 |
| recall    |    0.769483 |    0.784914 |     0.994906 |    0.98217  |     0.994682 |   0.97033  |
| f1        |    0.749956 |    0.767694 |     0.994513 |    0.981354 |     0.993974 |   0.967381 |
| number    | 3208        | 1518        | 24048        | 6618        | 10530        | nan        |
### xlnet-base-cased
#### System A


#### System B
