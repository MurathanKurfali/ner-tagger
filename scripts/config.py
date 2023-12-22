# the tag mapping is copied from: https://huggingface.co/datasets/Babelscape/multinerd .  It is weird, but I couldn't find this mapping
# in the dataset object.
tag_to_index_mapping = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-BIO": 9,
    "I-BIO": 10,
    "B-CEL": 11,
    "I-CEL": 12,
    "B-DIS": 13,
    "I-DIS": 14,
    "B-EVE": 15,
    "I-EVE": 16,
    "B-FOOD": 17,
    "I-FOOD": 18,
    "B-INST": 19,
    "I-INST": 20,
    "B-MEDIA": 21,
    "I-MEDIA": 22,
    "B-MYTH": 23,
    "I-MYTH": 24,
    "B-PLANT": 25,
    "I-PLANT": 26,
    "B-TIME": 27,
    "I-TIME": 28,
    "B-VEHI": 29,
    "I-VEHI": 30,
}

index_to_tag_mapping = {v: k for k, v in tag_to_index_mapping.items()}

### target entity/tag list for model B
model_b_entities =  ["PER", "ORG", "LOC", "DIS", "ANIM"]
model_b_tags = ["B-" + tag for tag in model_b_entities] + ["I-" + tag for tag in model_b_entities]
model_b_tag_indices = [tag_to_index_mapping[tag] for tag in model_b_tags]
