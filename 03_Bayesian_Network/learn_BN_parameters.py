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
    if len(z.shape) > 1:
        z = pd.DataFrame(z)
        res = z.apply(predict_y_from_z, axis=1)
        return res.to_numpy()
    else:
        if all([z[i] == 'nan' or pd.isnull(z[i]) or z[i] == 'entailment' for i in range(25)]):
            return 'entailment'
        elif any(z == 'contradiction'):
            return 'contradiction'
        elif any(z == 'neutral') and all(z != 'contradiction'):
            return 'neutral'
        else:
            raise ValueError(f"z can only have values 'entailment', 'contradiction', or 'neutral' but is {z}")

def most_likely_heuristic(clf, y_hat, not_nan, predicted_y, preds, log_prob):
    y_mapping = {clf[0].classes_[0]: 0, clf[0].classes_[1]: 1, clf[0].classes_[2]: 2}
    if predicted_y != y_hat:
        if y_hat == 'entailment':
            preds[not_nan] = 'entailment'
        elif y_hat == 'contradiction':
            best_i = 0
            best_swapping_cost = np.inf
            for i in range(25):
                if not_nan[i]:
                    swapping_cost = log_prob[i,y_mapping[preds[i]]] - log_prob[i,y_mapping['contradiction']]
                    if swapping_cost < best_swapping_cost:
                        best_i = i
                        best_swapping_cost = swapping_cost
            preds[best_i] = 'contradiction'
        elif y_hat == 'neutral':
            if predicted_y == 'contradiction':
                for i in np.where(pd.Series(preds) == 'contradiction')[0]:
                    replace_with_neutral = False
                    if not_nan[i]:
                        swapping_cost_ent = log_prob[i, y_mapping[preds[i]]] - log_prob[i][y_mapping['entailment']]
                        swapping_cost_neutr = log_prob[i, y_mapping[preds[i]]] - log_prob[i,y_mapping['neutral']]
                        if swapping_cost_neutr > swapping_cost_ent:
                            preds[i] = 'entailment'
                        else:
                            preds[i] = 'neutral'
                            replace_with_neutral = True
                if not replace_with_neutral:
                    best_i = 0
                    best_swapping_cost = np.inf
                    for i in range(25):
                        if not_nan[i]:
                            swapping_cost = log_prob[i, y_mapping[preds[i]]] - log_prob[i, y_mapping['neutral']]
                            if swapping_cost < best_swapping_cost:
                                best_i = i
                                best_swapping_cost = swapping_cost
                    preds[best_i] = 'neutral'
            elif predicted_y == 'entailment':
                best_i = 0
                best_swapping_cost = np.inf
                for i in range(25):
                    if not_nan[i]:
                        swapping_cost = log_prob[i, y_mapping["entailment"]] - log_prob[i, y_mapping["neutral"]]
                        if swapping_cost < best_swapping_cost:
                            best_i = i
                            best_swapping_cost = swapping_cost
                preds[best_i] = 'neutral'
    log_lik = 0
    for i in range(25):
        if not_nan[i]:
            log_lik += log_prob[i, y_mapping[preds[i]]]
    return preds, log_lik

def estimate_z_and_log_lik(clf, y_hat, data, not_nan):
    n = data.shape[0]
    preds = np.empty((y_hat.shape[0], 25), dtype=np.dtype('U25'))
    preds[:,:] = np.nan
    for i in range(25):
        preds[not_nan[:,i], i] = clf[i].predict(data[not_nan[:,i],:][:, cols[i]])
    predicted_y = predict_y_from_z(preds)
    print(f"Current Train Accuracy Before Adaptions: {np.mean(predicted_y == y_hat)}")
    log_prob = np.zeros((25, 3, n))
    for i in range(25):
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
    z_dev = np.empty((dev_data.shape[0], 25), dtype=np.dtype('U100'))
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
        for i in range(25):
            try:
                clf[i].fit(data[not_nan[:,i],][:,cols[i]], z[not_nan[:,i], i])
            except:
                models_not_updated += [i,]
                #print(f"Not all classes are present in the prediction so model {i} parameters are not updated")
                #z[not_nan[:,i], i][-6:] = ["contradiction"] * 2 + ["neutral"] * 2 + ["entailment"] * 2
                #clf[i].fit(data[not_nan[:,i],][:,cols[i]], z[not_nan[:,i], i])
            z_dev[not_nan_dev[:,i],i] = clf[i].predict(dev_data[not_nan_dev[:,i],][:,cols[i]])
        print(f"The following models are not updated because target y does not contain all classes: {models_not_updated}")
        y_pred = predict_y_from_z(z_dev)
        y_pred_test = predict_y_from_z(z)
        print(f"Current Train Accuracy After Adaptions: {np.mean(y_hat == y_pred_test)}")
        dev_acc += [np.mean(y_pred == y_dev), ]
        print(f"Current Dev Accuracy: {dev_acc[-1]}")


    return clf, z, log_lik, dev_acc

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

    amount_training_data = 480 # min 40, max 480
    batch_size= amount_training_data * 1000
    em_iter = 140
    mlp_iter = 100
    size_hidden_layers = (200, 200, 50, 30, 30)
    str_size_hidden = '_'.join([str(i) for i in size_hidden_layers])
    folder_name = 'MLP_Classifiers_' + str(amount_training_data) + 'k_training_' + str(em_iter) + '_iter_' + 'NN_size_' + str_size_hidden
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
    for i in range(25):
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
            res += [["entailment", ] * 25, ]
        elif label == "neutral":
            temp_ent = ["neutral", ] * 25
            temp_ent = ["entailment" if i in np.random.choice(range(25), size=np.random.randint(1,25), replace=False) else temp_ent[i] for i in range(25)]
            res += [temp_ent, ]
        elif label == "contradiction":
            temp_contr = list(np.random.choice(["entailment", "neutral"], size=25))
            temp_contr = ["contradiction" if i in np.random.choice(range(25), size=np.random.randint(1,25), replace=False) else temp_contr[i] for i in range(25)]
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

    print(datetime.datetime.now())
    print("Initialise 'nan' values for dev set")
    not_nan_dev = [None, ] * 25
    for i in range(25):
        not_nan_dev[i] = pd.Series([not x for x in pd.DataFrame(np.isnan(dev_data[:,cols[i]])).apply(any, axis=1)])
    not_nan_dev = np.array(not_nan_dev).T

    # Initialise MLP classifiers based on initial z values
    print(datetime.datetime.now())
    print("Initial Training of MLP classifiers with initial z values")
    for i in range(25):
        clf[i].fit(data[not_nan[:,i],][:,cols[i]], z[not_nan[:,i], i])

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
    for i in range(25):
        with open(folder_name + '/MLP_Classifier' + str(i) + '.pkl','wb') as f:
            pickle.dump(clf[i],f)




