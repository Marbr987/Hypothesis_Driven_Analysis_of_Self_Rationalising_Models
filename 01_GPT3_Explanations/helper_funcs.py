import random

import numpy as np
import pandas as pd

def single_row_to_string(row, include_label):
    s1 = row.Sentence1
    s2 = row.Sentence2
    expl = row.Explanation_1
    label= row.gold_label
    if include_label:
        return 'Statement: ' + s1 + '\nStatement: ' + s2 + '\nLabel: ' + label + '\nExplanation: ' + expl
    else:
        return 'Statement: ' + s1 + '\nStatement: ' + s2 + '\nExplanation: ' + expl

def prepare_examples(data, size_per_class=4, include_label=False, example_indices=False):
    if not example_indices:
        example_indices = list()
        for cat in ['neutral', 'contradiction', 'entailment']:
            example_indices += list(np.random.choice(data[data.gold_label == cat].index.values, size=size_per_class, replace=False))
            random.shuffle(example_indices)
    data = data.loc[example_indices]
    res = list()
    for row in data.itertuples():
        res += [single_row_to_string(row, include_label=include_label)]
    return '\n\n'.join(res)

def create_query(row, include_label=False):
    s1 = row.Sentence1
    s2 = row.Sentence2
    if include_label:
        return 'Statement: ' + s1 + '\nStatement: ' + s2 + '\nLabel: '
    else:
        return 'Statement: ' + s1 + '\nStatement: ' + s2 + '\nExplanation: '