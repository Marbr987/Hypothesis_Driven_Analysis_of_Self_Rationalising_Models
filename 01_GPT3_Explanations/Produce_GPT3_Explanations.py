import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import transformers
import os
import openai
import re
import time
from helper_funcs import prepare_examples, create_query
from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    #%%
    train1 = pd.read_csv('../Input_Data/e-SNLI/dataset/esnli_train_1.csv')
    train2 = pd.read_csv('../Input_Data/e-SNLI/dataset/esnli_train_2.csv')
    train = pd.concat([train1, train2])
    train = train[train.notnull().apply(all, axis=1)].reset_index()
    dev = pd.read_csv('../Input_Data/e-SNLI/dataset/esnli_dev.csv')
    dev = dev[dev.notnull().apply(all, axis=1)]
    test = pd.read_csv('../Input_Data/e-SNLI/dataset/esnli_test.csv')
    test = test[test.notnull().apply(all, axis=1)]

    i = 7
    index_range = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, dev.shape[0]]
    np.random.seed(index_range[i])
    dev_indices = range(index_range[i-1],index_range[i])
    prompts_instruct_GPT = [prepare_examples(train, size_per_class=2, style="customInstruct") + create_query(dev.loc[i], style="customInstruct") for i in dev_indices]

    dev_prepared = pd.DataFrame()
    dev_prepared['pairID'] = dev.loc[dev_indices].pairID
    dev_prepared['gold_standard_explanation'] = dev.loc[dev_indices].Explanation_1
    dev_prepared['gold_standard_label'] = dev.loc[dev_indices].gold_label
    dev_prepared['prompts_instruct_GPT'] = prompts_instruct_GPT

    labels = list()
    explanations = list()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = "org-qDPK8eQnXWK6wigTDVoWmU9B"
    for k in range(dev_prepared.shape[0]):
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt = dev_prepared['prompts_instruct_GPT'].iloc[k],
            temperature=0,
            max_tokens=58,
            top_p=1,
        )
        labels += [response.choices[0].text.split("\nExplanation: ")[0].strip(),]
        explanations += [response.choices[0].text.split("\nExplanation: ")[1], ]
        time.sleep(3)
    dev_prepared["pred_explanation"] = explanations
    dev_prepared["pred_label"] = labels
    dev_prepared.to_csv('prepared_data/GPT3_explanations' + str(i) + '.csv', sep=';')
