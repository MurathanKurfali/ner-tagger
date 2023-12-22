import json
from collections import defaultdict
from sklearn.metrics import f1_score
import os
import evaluate


# Function to read the gold data
def read_gold_data(file_path):
    gold_tags = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            gold_tags.append(entry['ner_tags'])
    return gold_tags


# Function to read predictions
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


# Calculate F1 score per tag
def calculate_scores_per_tag(true_labels, true_predictions):
    metric = evaluate.load("seqeval")
    final_results = defaultdict(list)

    for k, v in true_predictions.items():
        results = metric.compute(predictions=v, references=true_labels)
        # Unpack nested dictionaries
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"].append(v)
            else:
                final_results[key].append(value)
    print(final_results)
    mean_scores_dict = {key: sum(value) / len(value) for key, value in final_results.items()}

    return mean_scores_dict


# Main function to process everything
def main():
    gold_data_path = "../data/system_a/test.json"
    predictions_dir_path = '/home/murathan/PycharmProjects/rise/X'  # Update this with the correct path

    gold_tags = read_gold_data(gold_data_path)
    predictions = read_predictions(predictions_dir_path)
    scores_dict = calculate_scores_per_tag(gold_tags, predictions)


    # Restructure data
    restructured_data = {}
    for key, value in scores_dict.items():
        tag, metric = key.rsplit('_', 1)
        if metric == "number": continue
        if metric not in restructured_data:
            restructured_data[metric] = {}
        restructured_data[metric][tag] = value

    # Create Markdown table
    tags = sorted(set(key.rsplit('_', 1)[0] for key in scores_dict.keys()))
    print(tags)
    markdown_table = "| Metric | " + " | ".join(tags) + " |\n| --- |" + " --- |" * len(tags) + "\n"

    for metric in ['precision', 'recall', 'f1']:
        markdown_table += f"| {metric.capitalize()} | " + " | ".join( f"{restructured_data[metric].get(tag, 'N/A'):.3f}" for tag in tags) + " |\n"

    print(markdown_table)

if __name__ == "__main__":
    main()
