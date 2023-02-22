#%%
import pandas as pd
import datetime
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pickle

def predict_y_from_z(z):
    if len(z.shape) > 1:
        return np.apply_along_axis(predict_y_from_z, axis=1, arr=z)
    else:
        if all([z[i] == 'nan' or z[i] == 'neutral' for i in range(25)]):
            return 'neutral'
        elif any(z == 'contradiction'):
            return 'contradiction'
        else:
            return 'entailment'

def most_likely_heuristic(clf, y_hat, data, not_nan):
    y_mapping = {clf[0].classes_[0]: 0, clf[0].classes_[1]: 1, clf[0].classes_[2]: 2}
    preds = pd.Series([np.nan, ] * 25)
    for i in range(25):
        if not_nan[i]:
            preds[i] = clf[i].predict(data[cols[i]].reshape(1,-1))[0]
    predicted_y = predict_y_from_z(preds)
    log_prob = np.zeros((25,3))
    for i in range(25):
        if not_nan[i]:
            log_prob[i,:] = clf[i].predict_log_proba(data[cols[i]].reshape(1,-1))
    if predicted_y != y_hat:
        if y_hat == 'neutral':
            preds[not_nan] = 'neutral'
        elif y_hat == 'contradiction':
            best_i = 0
            best_swapping_cost = np.inf
            for i in range(25):
                if not_nan[i]:
                    swapping_cost = log_prob[i,y_mapping[preds[i]]] - log_prob[i,y_mapping['contradiction']]
                    if swapping_cost < best_swapping_cost:
                        best_i = i
            preds[best_i] = 'contradiction'
        elif y_hat == 'entailment':
            if predicted_y == 'contradiction':
                for i in np.where(pd.Series(preds) == 'contradiction'):
                    i = i[0]
                    replace_with_entailment = False
                    if not_nan[i]:
                        swapping_cost_ent = log_prob[i, y_mapping[preds[i]]] - log_prob[i][y_mapping['entailment']]
                        swapping_cost_neutr = log_prob[i, y_mapping[preds[i]]] - log_prob[i,y_mapping['neutral']]
                        if swapping_cost_neutr > swapping_cost_ent:
                            preds[i] = 'entailment'
                            replace_with_entailment = True
                        else:
                            preds[i] = 'neutral'
                if not replace_with_entailment:
                    best_i = 0
                    best_swapping_cost = np.inf
                    for i in range(25):
                        if not_nan[i]:
                            swapping_cost = log_prob[i, y_mapping[preds[i]]] - log_prob[i, y_mapping['entailment']]
                            if swapping_cost < best_swapping_cost:
                                best_i = i
                    preds[best_i] = 'entailment'
            elif predicted_y == 'neutral':
                best_i = 0
                best_swapping_cost = np.inf
                for i in range(25):
                    if not_nan[i]:
                        swapping_cost = log_prob[i, y_mapping[preds[i]]] - log_prob[i, y_mapping['entailment']]
                        if swapping_cost < best_swapping_cost:
                            best_i = i
                preds[best_i] = 'entailment'
    log_lik = 0
    for i in range(25):
        if not_nan[i]:
            log_lik += log_prob[i, y_mapping[preds[i]]]
    return preds, log_lik

def estimate_z_and_log_lik(clf, y_hat, data, not_nan):
    res = list()
    log_lik = 0
    for i in range(data.shape[0]):
        preds, cur_log_lik = most_likely_heuristic(clf, y_hat[i], data[i,:], not_nan[i,:])
        res += [preds, ]
        log_lik += cur_log_lik
    return np.array(res), log_lik

def em_algorithm(clf, y_hat, data, not_nan, iter):
    log_lik = list()
    for k in range(iter):
        print(datetime.datetime.now())
        print(f"EM-Algorithm Iteration {k}")
        # E-Step
        z, cur_log_lik = estimate_z_and_log_lik(clf, y_hat, data, not_nan)
        log_lik += [cur_log_lik, ]
        # M-Step
        for i in range(25):
            clf[i].fit(data[not_nan[:,i],][:,cols[i]], z[not_nan[:,i], i])

    return clf, z, log_lik

if __name__ == "__main__":
    #%%
    train1 = pd.read_csv('../Input_Data/e-SNLI/dataset/esnli_train_1.csv')
    train2 = pd.read_csv('../Input_Data/e-SNLI/dataset/esnli_train_2.csv')
    train = pd.concat([train1, train2])
    train = train[train.notnull().apply(all, axis=1)]
    dev = pd.read_csv('../Input_Data/e-SNLI/dataset/esnli_dev.csv')
    dev = dev[dev.notnull().apply(all, axis=1)]
    test = pd.read_csv('../Input_Data/e-SNLI/dataset/esnli_test.csv')
    test = test[test.notnull().apply(all, axis=1)]

    #data_prepared = pd.read_pickle("../02_Extract_Subphrases/prepared_data/subphrase_vectors_dev.pkl")
    data_prepared = pd.read_csv("../02_Extract_Subphrases/prepared_data/subphrase_vectors_20train.csv", sep=";")
    for i in np.arange(40,420,20):
        cur_data_prepared = pd.read_csv("../02_Extract_Subphrases/prepared_data/subphrase_vectors_" + str(i) + "train.csv", sep=";")
        data_prepared = data_prepared.append(cur_data_prepared)
    original_dataset = train

    #data = pd.DataFrame([vec for vec in data_prepared.vecs if vec is not None])
    data_prepared = data_prepared.iloc[:,1:]
    data = data_prepared
    data_index = data.iloc[:,0]
    data = data.iloc[:,1:].to_numpy().astype(float)

    original_dataset = original_dataset.set_index('pairID')
    data_prepared = data_prepared.set_index('0')

    # Initialise MLP Classifier
    print(datetime.datetime.now())
    print("Initialise MLP Classifier")
    clf = list()
    for i in range(25):
        clf += [MLPClassifier(hidden_layer_sizes=(25, 25, 10)), ]

    # Initialise z and y_hat
    print(datetime.datetime.now())
    print("Initialise z and y_hat")
    res = list()
    y_hat = list()
    for ind in data_index:
        label = original_dataset.loc[ind].gold_label
        y_hat += [label, ]
        if label == "neutral":
            res += [["neutral", ] * 25, ]
        elif label == "entailment":
            temp_ent = ["neutral", ] * 25
            temp_ent = ["entailment" if i in np.random.choice(range(25), size=np.random.randint(1,25), replace=False) else temp_ent[i] for i in range(25)]
            res += [temp_ent, ]
        else:
            temp_contr = list(np.random.choice(["entailment", "neutral"], size=25))
            temp_contr = ["contradiction" if i in np.random.choice(range(25), size=np.random.randint(1,25), replace=False) else temp_contr[i] for i in range(25)]
            res += [temp_contr, ]
    z = np.array(res)
    y_hat = np.array(y_hat)

    # Prepare colum indices
    indices = [[0,1500], [0,1800], [0,2100], [0,2400], [0,2700],
               [300,1500], [300,1800], [300,2100], [300,2400], [300,2700],
               [600,1500], [600,1800], [600,2100], [600,2400], [600,2700],
               [900,1500], [900,1800], [900,2100], [900,2400], [900,2700],
               [1200,1500], [1200,1800], [1200,2100], [1200,2400], [1200,2700]]

    # Initialise colulmn indices and "nan" values if information (e.g. location of sentence) is not detected
    print(datetime.datetime.now())
    print("Initialise column indices and 'nan' values")
    not_nan = [None, ] * 25
    cols = [None, ] * 25
    for i in range(25):
        cols[i] = list(range(indices[i][0], indices[i][0]+300)) + list(range(indices[i][1],indices[i][1]+300))
        not_nan[i] = pd.Series([not x for x in pd.DataFrame(np.isnan(data[:,cols[i]])).apply(any, axis=1)])
    not_nan = np.array(not_nan).T

    # Initialise MLP classifiers based on initial z values
    print(datetime.datetime.now())
    print("Initial Training of MLP classifiers with initial z values")
    for i in range(25):
        clf[i].fit(data[not_nan[:,i],][:,cols[i]], z[not_nan[:,i], i])

    # Run EM-algorithm
    clf, z, log_lik = em_algorithm(clf, y_hat, data, not_nan, iter=20)

    #Plot log likelihood
    plt.plot(log_lik)
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood EM-Algorithm')
    plt.savefig('log_likelihood.pdf')

    # Save MLP classifiers
    for i in range(25):
        with open('MLP_Classifiers/MLP_Classifier' + str(i) + '.pkl','wb') as f:
            pickle.dump(clf[i],f)




