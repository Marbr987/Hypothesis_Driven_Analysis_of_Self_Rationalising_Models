#%%
import pandas as pd
import datetime
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

def predict_y_from_z(z):

    # Iterate through each row if z is a matrix
    if len(z.shape) > 1:
        z = pd.DataFrame(z)
        res = z.apply(predict_y_from_z, axis=1)
        return res.to_numpy()

    # For each single line perform the following:
    else:
        # If any z is 'contradiction' -> output class 'contradiction'
        if any(z == 'contradiction'):
            return 'contradiction'
        # Else if all non-mixed hidden variables are 'entailment' an -> output class 'entailment'
        elif all([z[i] == 'nan' or pd.isnull(z[i]) or z[i] == 'entailment' for i in non_mixed_pairs_indices]):
            return 'entailment'
        # Else output class 'neutral'
        elif any(z == 'neutral'):
            return 'neutral'
        else:
            raise ValueError(f"z can only have values 'entailment', 'contradiction', or 'neutral' but is {z}")

def most_likely_heuristic(clf, y_hat, not_nan, predicted_y, preds, log_prob):

    y_mapping = {clf[0].classes_[0]: 0, clf[0].classes_[1]: 1, clf[0].classes_[2]: 2}
    if predicted_y != y_hat:

        # Should have predicted 'entailment'
        if y_hat == 'entailment':
            # If class is entailment, all non-mixed-variables must be of class 'entailment'
            preds[[True if i in non_mixed_pairs_indices and not_nan[i] else False for i in range(len(use_z_values))]] = 'entailment'
            # No hidden variable can have value 'contradiction'
            if predicted_y == 'contradiction':
                for i in np.where(pd.Series(preds) == 'contradiction')[0]:
                    if not_nan[i]:
                        swapping_cost_ent = log_prob[i, y_mapping[preds[i]]] - log_prob[i][y_mapping['entailment']]
                        swapping_cost_neutr = log_prob[i, y_mapping[preds[i]]] - log_prob[i,y_mapping['neutral']]
                        if swapping_cost_neutr > swapping_cost_ent:
                            preds[i] = 'entailment'
                        else:
                            preds[i] = 'neutral'

        # Should have predicted contradiction -> change least expensive one to contradiction
        elif y_hat == 'contradiction':
            best_i = 0
            best_swapping_cost = np.inf
            for i in range(len(use_z_values)):
                if not_nan[i]:
                    swapping_cost = log_prob[i,y_mapping[preds[i]]] - log_prob[i,y_mapping['contradiction']]
                    if swapping_cost < best_swapping_cost:
                        best_i = i
                        best_swapping_cost = swapping_cost
            preds[best_i] = 'contradiction'

        # Should have predicted neutral
        elif y_hat == 'neutral':
            # If predicted class is 'contradiction' -> change all those variables to either 'neutral' or 'entailment'
            if predicted_y == 'contradiction':
                for i in np.where(pd.Series(preds) == 'contradiction')[0]:
                    if not_nan[i]:
                        swapping_cost_ent = log_prob[i, y_mapping[preds[i]]] - log_prob[i][y_mapping['entailment']]
                        swapping_cost_neutr = log_prob[i, y_mapping[preds[i]]] - log_prob[i,y_mapping['neutral']]
                        if swapping_cost_neutr > swapping_cost_ent:
                            preds[i] = 'entailment'
                        else:
                            preds[i] = 'neutral'
                if not any([preds[i] == 'neutral' for i in non_mixed_pairs_indices]):
                    best_i = 0
                    best_swapping_cost = np.inf
                    for i in non_mixed_pairs_indices:
                        if not_nan[i]:
                            swapping_cost = log_prob[i, y_mapping[preds[i]]] - log_prob[i, y_mapping['neutral']]
                            if swapping_cost < best_swapping_cost:
                                best_i = i
                                best_swapping_cost = swapping_cost
                    preds[best_i] = 'neutral'
            elif predicted_y == 'entailment':
                best_i = 0
                best_swapping_cost = np.inf
                for i in non_mixed_pairs_indices:
                    if not_nan[i]:
                        swapping_cost = log_prob[i, y_mapping[preds[i]]] - log_prob[i, y_mapping['neutral']]
                        if swapping_cost < best_swapping_cost:
                            best_i = i
                            best_swapping_cost = swapping_cost
                preds[best_i] = 'neutral'
    log_lik = 0
    for i in range(len(use_z_values)):
        if not_nan[i]:
            log_lik += log_prob[i, y_mapping[preds[i]]]
    return preds, log_lik

def estimate_z_and_log_lik(clf, y_hat, data, not_nan):
    n = data.shape[0]
    preds = np.empty((y_hat.shape[0], len(use_z_values)), dtype=np.dtype('U25'))
    preds[:,:] = np.nan
    for i in range(len(use_z_values)):
        preds[not_nan[:,i], i] = clf[i].predict(data[not_nan[:,i],:][:, cols[i]])
    predicted_y = predict_y_from_z(preds)
    print(f"Current Train Accuracy Before Adaptions: {np.mean(predicted_y == y_hat)}")
    log_prob = np.zeros((len(use_z_values), 3, n))
    for i in range(len(use_z_values)):
        log_prob[i, :, not_nan[:,i]] = clf[i].predict_log_proba(data[not_nan[:,i],:][:, cols[i]])
    res = tuple(map(lambda y_hat, not_nan, predicted_y, preds, log_prob: most_likely_heuristic(clf, y_hat, not_nan, predicted_y, preds, log_prob),
                    (y_hat[i] for i in range(n)),
                    (not_nan[i,:] for i in range(n)),
                    (predicted_y[i] for i in range(n)),
                    (preds[i,:] for i in range(n)),
                    (log_prob[:,:,i] for i in range(n))))
    log_lik = sum([res[i][1] for i in range(n)])
    res = [res[i][0] for i in range(n)]
    return np.array(res), log_lik


def em_algorithm(clf, y_hat, data, not_nan, iter, dev_data, y_dev, not_nan_dev):
    log_lik = list()
    dev_acc = list()
    z_dev = np.empty((dev_data.shape[0], len(use_z_values)), dtype=np.dtype('U25'))
    z_dev[:,:] = np.nan
    for k in range(iter):
        print("======================================")
        print(datetime.datetime.now())
        print(f"EM-Algorithm Iteration {k+1}")
        # E-Step
        print(datetime.datetime.now())
        print("E-Step")
        z, cur_log_lik = estimate_z_and_log_lik(clf, y_hat, data, not_nan)
        log_lik += [cur_log_lik, ]
        print(f'Current log likelihood: {cur_log_lik}')
        # M-Step
        print(datetime.datetime.now())
        print("M-Step")
        models_not_updated = list()
        for i in range(len(use_z_values)):
            try:
                clf[i].fit(data[not_nan[:,i],][:,cols[i]], z[not_nan[:,i], i])
            except:
                models_not_updated += [i,]
            z_dev[not_nan_dev[:,i],i] = clf[i].predict(dev_data[not_nan_dev[:,i],][:,cols[i]])
        if len(models_not_updated) > 0:
            print(f"The following models are not updated because target y does not contain all classes: {models_not_updated}")
        y_pred_dev = predict_y_from_z(z_dev)
        y_pred_train = predict_y_from_z(z)
        print(f"Current Train Accuracy After Adaptions: {np.mean(y_hat == y_pred_train)}")
        dev_acc += [np.mean(y_pred_dev == y_dev), ]
        print(f"Current Dev Accuracy: {dev_acc[-1]}")


    return clf, z, log_lik, dev_acc

if __name__ == "__main__":

    train1 = pd.read_csv('../Input_Data/e-SNLI/dataset/esnli_train_1.csv')
    train2 = pd.read_csv('../Input_Data/e-SNLI/dataset/esnli_train_2.csv')
    train = pd.concat([train1, train2])
    train = train[train.notnull().apply(all, axis=1)]
    dev = pd.read_csv('../Input_Data/e-SNLI/dataset/esnli_dev.csv')
    dev = dev[dev.notnull().apply(all, axis=1)]
    test = pd.read_csv('../Input_Data/e-SNLI/dataset/esnli_test.csv')
    test = test[test.notnull().apply(all, axis=1)]

    amount_training_data = 480 # min 40, max 480
    batch_size= amount_training_data * 1000
    em_iter = 40
    mlp_iter = 100
    size_hidden_layers = (200, 200, 50, 50, 30, 30)
    str_size_hidden = '_'.join([str(i) for i in size_hidden_layers])
    use_z_values = (0,6,12,18,24)
    # Indices in terms of z for all hidden variables that are not mixed, e.g. Subject1-Subject2, Verb1-Verb2, etc.
    non_mixed_pairs_indices = [i for i in range(len(use_z_values)) if use_z_values[i] in (0,6,12,18,24)]

    folder_name = 'vers2/MLP_Classifiers_' + str(amount_training_data) + 'k_training_' + str(em_iter) + '_iter_' + 'NN_size_' + str_size_hidden
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    data_prepared = pd.read_csv("../02_Extract_Subphrases/prepared_data/subphrase_vectors_20train.csv", sep=";")
    for i in np.arange(40, amount_training_data + 1, 20):
        cur_data_prepared = pd.read_csv("../02_Extract_Subphrases/prepared_data/subphrase_vectors_" + str(i) + "train.csv", sep=";")
        data_prepared = data_prepared.append(cur_data_prepared)
    original_dataset = train
    original_dev = dev

    dev_data_prepared = pd.read_csv("../02_Extract_Subphrases/prepared_data/subphrase_vectors_dev.csv", sep=";")
    dev_data_prepared = dev_data_prepared.iloc[:,1:]
    dev_data = dev_data_prepared
    dev_data_index = dev_data.iloc[:,0]
    dev_data = dev_data.iloc[:,1:].to_numpy().astype(float)

    original_dev = original_dev.set_index('pairID')
    dev_data_prepared = dev_data_prepared.set_index('0')

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
    for i in range(len(use_z_values)):
        clf += [MLPClassifier(hidden_layer_sizes=size_hidden_layers,
                              batch_size=batch_size,
                              learning_rate_init=0.001,
                              max_iter=mlp_iter,
                              random_state=1234,
                              verbose=False,
                              warm_start=True,
                              early_stopping=True,
                              validation_fraction=0.2,
                              n_iter_no_change=10), ]

    # Initialise z and y_hat
    print(datetime.datetime.now())
    print("Initialise z and y_hat")
    res = list()
    y_hat = list()
    for ind in data_index:
        label = original_dataset.loc[ind].gold_label
        y_hat += [label, ]
        if label == "entailment":
            temp_ent = ["entailment" if i in non_mixed_pairs_indices else "neutral" for i in range(len(use_z_values))]
            res += [temp_ent, ]
        elif label == "neutral":
            temp_neutr = [np.random.choice(["entailment", "neutral"], 1)[0] if i in non_mixed_pairs_indices else "neutral" for i in range(len(use_z_values))]
            if any(temp_neutr[i] == "neutral" for i in non_mixed_pairs_indices):
                j = np.random.choice(non_mixed_pairs_indices, 1)[0]
                temp_neutr[j] = "neutral"
            res += [temp_neutr, ]
        elif label == "contradiction":
            temp_contr = np.random.choice(["entailment", "neutral", "contradiction"], size=len(use_z_values)).tolist()
            if all([temp_contr != "contradiction" for i in range(len(use_z_values))]):
                j = np.random.choice(non_mixed_pairs_indices, 1)[0]
                temp_contr[j] = "contradiction"
            res += [temp_contr, ]
        else:
            raise ValueError(f"label must either be entailment, contradiction, or neutral. Not {label}")
    z = np.array(res)
    y_hat = np.array(y_hat)

    # Initialise y_dev
    print(datetime.datetime.now())
    print("Initialise y_dev")
    y_dev = np.array([original_dev.loc[ind].gold_label for ind in dev_data_index])

    # Prepare colum indices
    indices = np.array([[0,1500], [0,1800], [0,2100], [0,2400], [0,2700],
               [300,1500], [300,1800], [300,2100], [300,2400], [300,2700],
               [600,1500], [600,1800], [600,2100], [600,2400], [600,2700],
               [900,1500], [900,1800], [900,2100], [900,2400], [900,2700],
               [1200,1500], [1200,1800], [1200,2100], [1200,2400], [1200,2700]])
    indices = indices[use_z_values,:].tolist()

    # Initialise column indices and "nan" values if information (e.g. location of sentence) is not detected
    print(datetime.datetime.now())
    print("Initialise column indices and 'nan' values")
    not_nan = [None, ] * len(use_z_values)
    cols = [None, ] * len(use_z_values)
    for i in range(len(use_z_values)):
        cols[i] = list(range(indices[i][0], indices[i][0]+300)) + list(range(indices[i][1],indices[i][1]+300))
        not_nan[i] = pd.Series([not x for x in pd.DataFrame(np.isnan(data[:,cols[i]])).apply(any, axis=1)])
    not_nan = np.array(not_nan).T

    print(datetime.datetime.now())
    print("Initialise 'nan' values for dev set")
    not_nan_dev = [None, ] * len(use_z_values)
    for i in range(len(use_z_values)):
        not_nan_dev[i] = pd.Series([not x for x in pd.DataFrame(np.isnan(dev_data[:,cols[i]])).apply(any, axis=1)])
    not_nan_dev = np.array(not_nan_dev).T

    # Initialise MLP classifiers based on initial z values
    print(datetime.datetime.now())
    print("Initial Training of MLP classifiers with initial z values")
    z_dev = np.empty((dev_data.shape[0], len(use_z_values)), dtype=np.dtype('U25'))
    z_dev[:,:] = np.nan
    for i in range(len(use_z_values)):
        clf[i].fit(data[not_nan[:,i],][:,cols[i]], z[not_nan[:,i], i])
        z_dev[not_nan_dev[:,i],i] = clf[i].predict(dev_data[not_nan_dev[:,i],][:,cols[i]])
    y_pred_dev = predict_y_from_z(z_dev)
    print(f"Dev Accuracy with initial z values: {np.mean(y_pred_dev == y_dev)}")

    # Run EM-algorithm
    clf, z, log_lik, dev_acc = em_algorithm(clf,
                                            y_hat,
                                            data,
                                            not_nan,
                                            iter=em_iter,
                                            dev_data=dev_data,
                                            y_dev = y_dev,
                                            not_nan_dev=not_nan_dev)

    # Plot log likelihood
    plt.figure(1)
    plt.plot(log_lik)
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood EM-Algorithm')
    plt.savefig(folder_name + '/log_likelihood.pdf')

    # Plot dev accuracy
    plt.figure(2)
    plt.plot(dev_acc)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Dev Set EM-Algorithm')
    plt.savefig(folder_name + '/dev_accuracy.pdf')

    # Save MLP classifiers
    for i in range(len(use_z_values)):
        with open(folder_name + '/MLP_Classifier' + str(i) + '.pkl','wb') as f:
            pickle.dump(clf[i],f)
