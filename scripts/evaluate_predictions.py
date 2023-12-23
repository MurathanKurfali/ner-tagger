import json
import argparse
from collections import defaultdict
import os
import evaluate
import pandas as pd

# Read the gold annotations from the jsonl file
def read_gold_data(file_path):
    gold_tags = []
    gold_tokens = []  # To store tokens
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            gold_tags.append(entry['ner_tags'])
            gold_tokens.append(entry['tokens'])  # Extract tokens
    return gold_tags, gold_tokens


def filter_unseen_tokens(training_tokens, training_tags,test_tokens, test_tags, test_predictions):
    unseen_tokens_predictions = defaultdict(list)

    # Flatten the training tokens list for easier membership checking

    training_tokens_flat = [token for instance in training_tokens for token in instance]
    training_tags_flat = [tag for instance in training_tags for tag in instance]
    seen_token_tag_pairs = set(list(zip(training_tokens_flat, training_tags_flat)))

    # learn a masking to mask out the (tag,token) pairs that are in the training data too
    mask_out_predictions = []
    for tokens, tags in zip(test_tokens, test_tags):
        sentence_mask = []
        # for each (token,tag) in a sentence, check if they are in the training data
        for token, tag in zip(tokens, tags):
            if (token, tag) not in seen_token_tag_pairs:
                sentence_mask.append(True)
            else:
                sentence_mask.append(False)
        mask_out_predictions.append(sentence_mask)

    # apply the mask and leave the unseen tokens in the test data
    masked_test_tags = [[token for token, keep in zip(tokens, mask) if keep] for tokens, mask in zip(test_tags, mask_out_predictions)]
    masked_test_predictions = {}
    for prediction_file, predictions in test_predictions.items():
        masked_test_predictions[prediction_file] = [[token for token, keep in zip(tokens, mask) if keep] for tokens, mask in zip(predictions, mask_out_predictions)]

    return masked_test_tags, masked_test_predictions

# The fine-tuning process saves the predictions of the model in
# "predictions.txt" file.
def read_predictions(dir_path, transformer_model):
    predictions = defaultdict(list)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('predictions.txt') and transformer_model in root:
                with open(os.path.join(root, file), 'r') as f:
                    file_predictions = f.readlines()
                    for line in file_predictions:
                        predictions[f].append(line.strip().split())
    return predictions


# # Calculates scores for each tag using the 'evaluate' and seqeval library
def calculate_scores_per_tag(true_labels, true_predictions):
    metric = evaluate.load("seqeval")
    final_results = defaultdict(list)

    for k, v in true_predictions.items():
        results = metric.compute(predictions=v, references=true_labels)
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"].append(v)
            else:
                final_results[key].append(value)
    mean_scores_dict = {key: sum(value) / len(value) for key, value in final_results.items()}

    # Restructure the results to get more readable tables.
    structured_scores = {}
    for key, value in mean_scores_dict.items():
        tag, metric = key.rsplit('_', 1)
        if metric == "accuracy": continue
        if metric not in structured_scores:
            structured_scores[metric] = {}
        structured_scores[metric][tag] = value

    return structured_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER Tagging Evaluation')
    parser.add_argument('saved_model_dir', type=str, help='Directory of saved model')
    parser.add_argument('gold_data_dir', type=str, help='Directory of gold data')
    parser.add_argument('transformer_model', type=str, help='Transformer model name')
    parser.add_argument('only_unseen', type=bool, default=True, help='Transformer model name')
    parser.add_argument('output_format', type=str, default='html', nargs='?', choices=['csv', 'html', "markdown"],
                        help='Output format for the results (default: html)')
    args = parser.parse_args()

    for system_version in ["a", "b"]:
        predictions_dir_path = os.path.join(args.saved_model_dir, f"system_{system_version}")
        gold_data_path = f"{args.gold_data_dir}/system_{system_version}/test.json"

        test_tags, test_tokens = read_gold_data(gold_data_path)
        predictions = read_predictions(predictions_dir_path, args.transformer_model)
        run_count = len(predictions)
        assert run_count > 0, print("No predictions.txt is found. Make sure you entered the correct path/ transformer model name")
        output_file_name = f"system-{system_version}_{args.transformer_model}_{run_count}.{args.output_format}"

        if args.only_unseen:
            training_gold_data_path = gold_data_path.replace("test", "train")
            training_tags, training_tokens = read_gold_data(training_gold_data_path)
            test_tags, predictions = filter_unseen_tokens(training_tokens,training_tags,test_tokens, test_tags, predictions)
            output_file_name = output_file_name.replace(f".{args.output_format}", f"-unseen.{args.output_format}")

        scores_dict = calculate_scores_per_tag(test_tags, predictions)

        df = pd.DataFrame.from_dict(scores_dict, orient='index').T

        # save the DataFrame in the specified format
        if args.output_format == 'csv':
            df.to_csv(output_file_name, index=True)
        elif args.output_format == 'html':
            df.to_html(output_file_name, index=True)
        elif args.output_format == 'markdown':
            df = df.T
            markdown_table = df.to_markdown(index=True)
            # Write to file
            with open(output_file_name, 'w') as file:
                file.write(markdown_table)
        print(f"The results for system {system_version} are saved to {output_file_name}")