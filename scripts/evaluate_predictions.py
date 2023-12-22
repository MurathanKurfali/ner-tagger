import json
import argparse
from collections import defaultdict
import os
import evaluate
import pandas as pd

# Read the gold annotations from the jsonl file
def read_gold_data(file_path):
    gold_tags = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            gold_tags.append(entry['ner_tags'])
    return gold_tags

# The fine-tuning process saves the predictions of the model in
# "predictions.txt" file.
def read_predictions(dir_path):
    predictions = defaultdict(list)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('predictions.txt'):
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
    parser.add_argument('output_format', type=str, default='html', nargs='?', choices=['csv', 'html', "markdown"],
                        help='Output format for the results (default: html)')
    args = parser.parse_args()

    for system_version in ["a", "b"]:
        predictions_dir_path = os.path.join(args.saved_model_dir, f"system_{system_version}")
        gold_data_path = f"{args.gold_data_dir}/system_{system_version}/test.json"

        gold_tags = read_gold_data(gold_data_path)
        predictions = read_predictions(predictions_dir_path)
        run_count = len(predictions)
        scores_dict = calculate_scores_per_tag(gold_tags, predictions)
        output_file_name = f"system-{system_version}_{args.transformer_model}_{run_count}.{args.output_format}"

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