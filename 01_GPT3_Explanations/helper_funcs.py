import random

import numpy as np
import pandas as pd

def single_row_to_string(row, style):
    s1 = row.Sentence1
    s2 = row.Sentence2
    expl = row.Explanation_1
    label= row.gold_label
    label_map = {"entailment": "Yes",
                 "neutral": "Maybe",
                 "contradiction": "No"}
    if style=="GPT-3" or style=="customInstruct":
        return ('Premise: ' + s1 + \
                '\nHypothesis: ' + s2 + \
                '\nLabel: ' + label + \
                '\nExplanation: ' + expl + '\n###\n')
    elif style=="ZhengEtAl":
        return (s1 +\
               '\nQuestion: Is ' + s2.lower().replace(".", "") +\
               '?\nAnswer: ' + label_map[label] +\
               '\nReason: ' + expl + '\n###\n')
    elif style=="customInstruct_alt4":
        return ('Premise: ' + s1 + \
                '\nHypothesis: ' + s2 + \
                '\nQuestion: Is the hypothesis true based on the premise?' + \
                '?\nAnswer: ' + label_map[label] + ', because ' + expl.lower() + '\n###\n')

def prepare_examples(data, size_per_class=4, style="GPT-3", example_indices=False):
    if not example_indices:
        example_indices = list()
        for cat in ['neutral', 'contradiction', 'entailment']:
            example_indices += list(np.random.choice(data[data.gold_label == cat].index.values, size=size_per_class, replace=False))
            random.shuffle(example_indices)
    data = data.loc[example_indices]
    res = list()
    for row in data.itertuples():
        res += [single_row_to_string(row, style=style)]
    if style=="GPT-3":
        return ''.join(res)
    elif style=="ZhengEtAl" or style=="customInstruct_alt4":
        return 'Answer the Question and provide a reason why the answer is correct.\n\n' + ''.join(res)
    elif style=="customInstruct":
        alt = 1
        if alt == 1:
            return 'Classify into entailment, neutral, or contradiction and justify the decision.\n\n' + ''.join(res)
        elif alt == 2:
            return 'Perform natural language inference.\n\n' + ''.join(res)
        elif alt == 3:
            return 'Does the hypothesis contradict the premise?\n\n' + ''.join(res)

def create_query(row, style="GPT-3"):
    s1 = row.Sentence1
    s2 = row.Sentence2
    if style=="GPT-3" or style=="customInstruct":
        return ('Premise: ' + s1 + \
                '\nHypothesis: ' + s2 + \
                '\nLabel: ')
    elif style=="ZhengEtAl":
        return (s1 +\
            '\nQuestion: Is ' + s2.lower().replace(".", "") +\
            '?\nAnswer: ')
    elif style=="customInstruct_alt4":
        return ('Premise: ' + s1 + \
                '\nHypothesis: ' + s2 + \
                '\nQuestion: Is the hypothesis true based on the premise?' + \
                '\nAnswer: ')