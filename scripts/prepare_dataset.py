import json

from datasets import load_dataset
import argparse
import os
from config import model_b_tags, index_to_tag_mapping
from tqdm import tqdm


def check_output_dir(output_dir):
    """
    Check if the output directory exists, and create it if it doesn't.

    Parameters:
    output_dir (str): The path to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")


def rename_and_filter_tags(batch, is_model_b):
    actual_tags = []
    for example in batch["ner_tags"]:
        if is_model_b:
            actual_tags.append( [index_to_tag_mapping[tag] if index_to_tag_mapping[tag] in model_b_tags else "O" for tag in example])
        else:
            actual_tags.append( [index_to_tag_mapping[tag] for tag in example])
    batch["ner_tags"] = actual_tags
    return batch


def filter_dataset(dataset, lang, is_model_b):
    # filter the dataset by language
    filtered_dataset = {split: data.filter(lambda example: example['lang'] == lang) for split, data in dataset.items()}

    # Further filter the dataset by the specified list of tags
    # This assumes that the dataset has a structure where tags are in a list
    filtered_dataset = {split: data.map(lambda batch: rename_and_filter_tags(batch, is_model_b),
                                        batched=True,
                                        batch_size=1000,
                                        num_proc=4,
                                        load_from_cache_file=False  # Disable caching
                                        ) for split, data in filtered_dataset.items()}

    # Output the filtered dataset
    return filtered_dataset


def save_dataset_as_json(dataset, output_dir):
    """
    Save the filtered dataset in a csv format suitable for fine-tuning.
    The files in the output will be overwritten, no control for that.
    Each line in the file will contain a word and its tag, separated by a comma.
    """
    check_output_dir(output_dir)
    for split, data in dataset.items():
        with open(f"{output_dir}/{split}.json", "w", encoding="utf-8") as file:
            for example in tqdm(data, desc=f"Saving {split} set"):
                json_line = {
                    "tokens": example['tokens'],
                    "ner_tags": example['ner_tags']
                }
                file.write(json.dumps(json_line) + '\n')

            print(f"{split} file is saved to {output_dir}/{split}.json with {len(data)} examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and filter a dataset from Hugging Face.')

    # Add arguments to the parser
    parser.add_argument('--dataset_name', type=str, default="Babelscape/multinerd", help='Name of the dataset to load')
    parser.add_argument('--language', type=str, default="en", help='Language to filter the dataset by')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save the output file')
    parser.add_argument('--tag-set', choices=['A', 'B', 'a', 'b'], default='A', help='Tag to filter the dataset by')

    # Parse the arguments
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)
    assert "ner_tags" in dataset.column_names["train"], print("The provided dataset is not a NER dataset")
    if args.tag_set == "B":
        is_model_b = True
        output_dir = os.path.join(args.output_dir, "system_b")
    else:
        is_model_b = False
        output_dir = os.path.join(args.output_dir, "system_a")

    filtered_dataset = filter_dataset(dataset, args.language, is_model_b)
    save_dataset_as_json(filtered_dataset, output_dir)
