
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
python scripts/evaluate_predictions.py "saved_models" <data_dir> <target-model> [<only-unseen>] [<output-format>]
```
This script calculates precision, recall, and F1-score for each tag. Additionally, the script supports evaluation only on those (token, tag) pairs which weren't seen during training, to test the model's generalization abilities. By default, the results are saved as a Markdown table. However, you have the option to save the results in CSV, HTML, or Markdown formats.

### Quick Start Example

The following example demonstrates the preparation of the dataset with the tag set "A," followed by running the main script with the default parameters.

```bash
python prepare_dataset.py --tag-set "a"
./run.sh "a" "bert-base-cased" "False"
python scripts/evaluate_predictions.py "saved_models" "data" True "bert-base-cased" 
```

### 4) Results

Below are the results for both configurations using bert-base-cased, roberta-base, and xlnet-base-cased models. These results are averaged over four runs and include both the entire test set and only the unseen (token, tag) pairs.
## 4.1 Bert-base-cased
### 4.1.1 On Test Set
#### 4.1.1.1 System A
|           |        ANIM |       BIO |       CEL |         DIS |        EVE |        FOOD |      INST |          LOC |      MEDIA |      MYTH |         ORG |          PER |       PLANT |       TIME |      VEHI |    overall |
|:----------|------------:|----------:|----------:|------------:|-----------:|------------:|----------:|-------------:|-----------:|----------:|------------:|-------------:|------------:|-----------:|----------:|-----------:|
| precision |    0.730958 |  0.42372  |  0.748029 |    0.744328 |   0.954162 |    0.653681 |  0.688988 |     0.994802 |   0.962849 |  0.796495 |    0.979856 |     0.993224 |    0.635877 |   0.826381 |  0.862262 |   0.941031 |
| recall    |    0.778367 |  0.75     |  0.829268 |    0.796443 |   0.975142 |    0.689488 |  0.770833 |     0.994677 |   0.972162 |  0.84375  |    0.984436 |     0.995204 |    0.730425 |   0.846886 |  0.921875 |   0.955059 |
| f1        |    0.753792 |  0.538377 |  0.786287 |    0.769469 |   0.964524 |    0.671099 |  0.725046 |     0.994739 |   0.967441 |  0.81867  |    0.982138 |     0.994213 |    0.679704 |   0.836433 |  0.890858 |   0.94799  |
| number    | 3208        | 16        | 82        | 1518        | 704        | 1132        | 24        | 24048        | 916        | 64        | 6618        | 10530        | 1788        | 578        | 64        | nan        |
#### 4.1.1.2 System B
|           |        ANIM |         DIS |          LOC |         ORG |          PER |    overall |
|:----------|------------:|------------:|-------------:|------------:|-------------:|-----------:|
| precision |    0.731486 |    0.751412 |     0.994121 |    0.98054  |     0.993268 |   0.964451 |
| recall    |    0.769483 |    0.784914 |     0.994906 |    0.98217  |     0.994682 |   0.97033  |
| f1        |    0.749956 |    0.767694 |     0.994513 |    0.981354 |     0.993974 |   0.967381 |
| number    | 3208        | 1518        | 24048        | 6618        | 10530        | nan        |

### 4.1.2 Only on the unseen (token,tag) Pairs
##### 4.1.2.1 System A

|           |       ANIM |      BIO |      CEL |        DIS |        EVE |       FOOD |      INST |         LOC |      MEDIA |      MYTH |         ORG |         PER |      PLANT |       TIME |      VEHI |    overall |
|:----------|-----------:|---------:|---------:|-----------:|-----------:|-----------:|----------:|------------:|-----------:|----------:|------------:|------------:|-----------:|-----------:|----------:|-----------:|
| precision |   0.631761 | 0.279167 | 0.620833 |   0.559386 |   0.844952 |   0.560985 |  0.68125  |    0.988916 |   0.953442 |  0.594444 |    0.947187 |    0.983963 |   0.525323 |   0.751609 |  0.866625 |   0.922308 |
| recall    |   0.596711 | 0.625    | 0.625    |   0.5      |   0.908333 |   0.462617 |  0.642857 |    0.991075 |   0.956422 |  0.6875   |    0.949838 |    0.989656 |   0.512552 |   0.567308 |  0.94     |   0.91576  |
| f1        |   0.613553 | 0.379762 | 0.612897 |   0.527662 |   0.875484 |   0.506628 |  0.647059 |    0.989993 |   0.954835 |  0.637255 |    0.948503 |    0.9868   |   0.5186   |   0.646459 |  0.90056  |   0.919021 |
| number    | 760        | 4        | 8        | 212        | 120        | 214        | 14        | 5266        | 436        | 16        | 1236        | 2852        | 478        | 104        | 50        | nan        |
##### 4.1.2.2 System B
|           |       ANIM |        DIS |         LOC |         ORG |         PER |    overall |
|:----------|-----------:|-----------:|------------:|------------:|------------:|-----------:|
| precision |   0.64109  |   0.598758 |    0.989661 |    0.946302 |    0.986185 |   0.953712 |
| recall    |   0.578496 |   0.471698 |    0.9906   |    0.947816 |    0.988254 |   0.943917 |
| f1        |   0.608103 |   0.527448 |    0.99013  |    0.947045 |    0.987214 |   0.948787 |
| number    | 758        | 212        | 5266        | 1236        | 2852        | nan        |

## 4.2 xlnet-base-cased
### 4.2.1 On Test Set

#### 4.2.1.1 System A
|           |        ANIM |       BIO |       CEL |         DIS |        EVE |        FOOD |      INST |          LOC |      MEDIA |      MYTH |         ORG |          PER |       PLANT |       TIME |      VEHI |    overall |
|:----------|------------:|----------:|----------:|------------:|-----------:|------------:|----------:|-------------:|-----------:|----------:|------------:|-------------:|------------:|-----------:|----------:|-----------:|
| precision |    0.733569 |  0.60964  |  0.742936 |    0.757945 |   0.952452 |    0.685616 |  0.550924 |     0.995505 |   0.964377 |  0.85567  |    0.978787 |     0.993135 |    0.639824 |   0.844199 |  0.768502 |   0.942931 |
| recall    |    0.801278 |  0.96875  |  0.804878 |    0.790843 |   0.981534 |    0.691696 |  0.604167 |     0.994719 |   0.972707 |  0.867188 |    0.983001 |     0.995916 |    0.738255 |   0.843426 |  0.84375  |   0.95657  |
| f1        |    0.765886 |  0.746981 |  0.772142 |    0.774022 |   0.966771 |    0.688537 |  0.574623 |     0.995112 |   0.968499 |  0.860848 |    0.980889 |     0.994523 |    0.685502 |   0.843794 |  0.803405 |   0.949701 |
| number    | 3208        | 16        | 82        | 1518        | 704        | 1132        | 24        | 24048        | 916        | 64        | 6618        | 10530        | 1788        | 578        | 64        | nan        |
#### 4.2.1.2 System B
|           |        ANIM |         DIS |          LOC |         ORG |          PER |    overall |
|:----------|------------:|------------:|-------------:|------------:|-------------:|-----------:|
| precision |    0.731532 |    0.764948 |     0.994718 |    0.977805 |     0.993037 |   0.964333 |
| recall    |    0.788653 |    0.801054 |     0.994407 |    0.981641 |     0.995347 |   0.972018 |
| f1        |    0.758973 |    0.782517 |     0.994562 |    0.979717 |     0.99419  |   0.96816  |
| number    | 3208        | 1518        | 24048        | 6618        | 10530        | nan        |

### 4.2.2. Only on the unseen (token,tag) Pairs
#### 4.2.2.1 System A

|           |       ANIM |      BIO |    CEL |        DIS |        EVE |       FOOD |      INST |         LOC |      MEDIA |      MYTH |         ORG |         PER |      PLANT |       TIME |      VEHI |    overall |
|:----------|-----------:|---------:|-------:|-----------:|-----------:|-----------:|----------:|------------:|-----------:|----------:|------------:|------------:|-----------:|-----------:|----------:|-----------:|
| precision |   0.647222 | 0.641667 | 0.6875 |   0.613172 |   0.835199 |   0.629299 |  0.541667 |    0.98966  |   0.958525 |  0.642424 |    0.935747 |    0.982642 |   0.504707 |   0.759939 |  0.805428 |   0.922577 |
| recall    |   0.641447 | 1        | 0.75   |   0.488208 |   0.929167 |   0.490654 |  0.392857 |    0.990505 |   0.951835 |  0.8125   |    0.953883 |    0.99176  |   0.514644 |   0.591346 |  0.87     |   0.919754 |
| f1        |   0.644176 | 0.759524 | 0.7125 |   0.543471 |   0.879671 |   0.550987 |  0.453089 |    0.990082 |   0.955131 |  0.713377 |    0.944718 |    0.987176 |   0.509612 |   0.664798 |  0.836325 |   0.921162 |
| number    | 760        | 4        | 8      | 212        | 120        | 214        | 14        | 5266        | 436        | 16        | 1236        | 2852        | 478        | 104        | 50        | nan        |
#### 4.2.2.2 System B
|           |       ANIM |       DIS |         LOC |         ORG |         PER |    overall |
|:----------|-----------:|----------:|------------:|------------:|------------:|-----------:|
| precision |   0.655537 |   0.5862  |    0.989093 |    0.930652 |    0.985182 |   0.95137  |
| recall    |   0.606201 |   0.46934 |    0.98984  |    0.94377  |    0.990182 |   0.945564 |
| f1        |   0.629896 |   0.52082 |    0.989464 |    0.93715  |    0.987673 |   0.948458 |
| number    | 758        | 212       | 5266        | 1236        | 2852        | nan        |

## 4.3 roberta-base
### 4.3.1 On Test Set

#### 4.3.1.1 System A
|           |        ANIM |       BIO |       CEL |         DIS |        EVE |        FOOD |      INST |          LOC |      MEDIA |      MYTH |         ORG |          PER |       PLANT |       TIME |      VEHI |    overall |
|:----------|------------:|----------:|----------:|------------:|-----------:|------------:|----------:|-------------:|-----------:|----------:|------------:|-------------:|------------:|-----------:|----------:|-----------:|
| precision |    0.727385 |  0.705808 |  0.759407 |    0.758737 |   0.946195 |    0.674453 |  0.756868 |     0.995442 |   0.949118 |  0.860845 |    0.980944 |     0.993702 |    0.652313 |   0.80445  |  0.84493  |   0.942881 |
| recall    |    0.782575 |  0.78125  |  0.853659 |    0.79249  |   0.972301 |    0.684629 |  0.833333 |     0.994365 |   0.976528 |  0.820312 |    0.983983 |     0.996344 |    0.743289 |   0.838235 |  0.929688 |   0.955576 |
| f1        |    0.753934 |  0.738777 |  0.803009 |    0.775065 |   0.959037 |    0.679402 |  0.793077 |     0.994903 |   0.962616 |  0.839814 |    0.982461 |     0.995021 |    0.694747 |   0.820918 |  0.885076 |   0.949184 |
| number    | 3208        | 16        | 82        | 1518        | 704        | 1132        | 24        | 24048        | 916        | 64        | 6618        | 10530        | 1788        | 578        | 64        | nan        |

#### 4.3.1.2 System B

|           |        ANIM |         DIS |          LOC |         ORG |          PER |    overall |
|:----------|------------:|------------:|-------------:|------------:|-------------:|-----------:|
| precision |    0.72808  |    0.758198 |     0.995422 |    0.979848 |     0.994076 |   0.96493  |
| recall    |    0.776496 |    0.792819 |     0.994615 |    0.984361 |     0.995916 |   0.971528 |
| f1        |    0.751444 |    0.775028 |     0.995018 |    0.982098 |     0.994995 |   0.968218 |
| number    | 3208        | 1518        | 24048        | 6618        | 10530        | nan        |

### 4.3.2 Only on the unseen (token,tag) Pairs
#### 4.3.2.1 System A
|           |       ANIM |      BIO |      CEL |        DIS |        EVE |       FOOD |      INST |         LOC |      MEDIA |      MYTH |         ORG |         PER |      PLANT |       TIME |      VEHI |    overall |
|:----------|-----------:|---------:|---------:|-----------:|-----------:|-----------:|----------:|------------:|-----------:|----------:|------------:|------------:|-----------:|-----------:|----------:|-----------:|
| precision |   0.665216 | 0.875    | 0.754167 |   0.651055 |   0.830598 |   0.603187 |  0.700893 |    0.989476 |   0.941684 |  0.767857 |    0.949845 |    0.985375 |   0.543354 |   0.727621 |  0.869757 |   0.927669 |
| recall    |   0.628289 | 1        | 0.8125   |   0.554245 |   0.895833 |   0.476636 |  0.75     |    0.99041  |   0.961009 |  0.71875  |    0.957524 |    0.992111 |   0.561715 |   0.576923 |  0.96     |   0.92277  |
| f1        |   0.646142 | 0.916667 | 0.768452 |   0.598101 |   0.861693 |   0.531636 |  0.72381  |    0.98994  |   0.951214 |  0.741667 |    0.953661 |    0.98873  |   0.552333 |   0.643362 |  0.911529 |   0.925209 |
| number    | 760        | 4        | 8        | 212        | 120        | 214        | 14        | 5266        | 436        | 16        | 1236        | 2852        | 478        | 104        | 50        | nan        |
#### 4.3.2.2 System B
|           |       ANIM |        DIS |         LOC |         ORG |         PER |    overall |
|:----------|-----------:|-----------:|------------:|------------:|------------:|-----------:|
| precision |   0.666633 |   0.629247 |    0.991911 |    0.941095 |    0.987944 |   0.955615 |
| recall    |   0.62467  |   0.537736 |    0.989651 |    0.955906 |    0.991059 |   0.949923 |
| f1        |   0.644896 |   0.579083 |    0.990779 |    0.948429 |    0.989498 |   0.95276  |
| number    | 758        | 212        | 5266        | 1236        | 2852        | nan        |

