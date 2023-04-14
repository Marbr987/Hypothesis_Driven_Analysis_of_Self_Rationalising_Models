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

def initialise_z(ignore_var):

    def temp_fun(ind):
        label = original_dataset.loc[ind].gold_label

        if label == "entailment":
            temp_ent = np.array(["neutral"] * len(use_z_values), dtype=np.dtype('U25'))
            ent_indices = list()
            for subphrase_indices in Sentence2_indices:
                ent_indices += np.random.choice(subphrase_indices, np.random.randint(1,len(subphrase_indices)+1), replace=False).tolist()
            temp_ent[ent_indices] = "entailment"
            return temp_ent

        elif label == "neutral":
            temp_neutr = [np.random.choice(["entailment", "neutral"], 1)[0] for i in range(len(use_z_values))]
            ent_mask = [any([temp_neutr[i] == 'entailment' for i in subphrase_indices]) for subphrase_indices in Sentence2_indices]
            if all(ent_mask):
                for i in Sentence2_indices[np.random.choice(np.where(ent_mask)[0], 1)[0]]:
                    temp_neutr[i] = "neutral"
            return temp_neutr

        elif label == "contradiction":
            temp_contr = np.random.choice(["entailment", "neutral", "contradiction"], size=len(use_z_values)).tolist()
            if all([temp_contr != "contradiction" for i in range(len(use_z_values))]):
                j = np.random.choice(range(len(use_z_values)), 1)[0]
                temp_contr[j] = "contradiction"
            return temp_contr

        else:
            raise ValueError(f"label must either be entailment, contradiction, or neutral. Not {label}")

    res = tuple(map(temp_fun, data_index))
    return np.array(res)

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
        # Else if all subphrases of sentence 2 are entailed by any subphrase of sentence 1 -> output class 'entailment'
        elif all([any([z[i] == 'entailment' for i in subphrase_indices]) or all([z[i] == 'nan' or pd.isnull(z[i]) for i in subphrase_indices]) for subphrase_indices in Sentence2_indices]):
            return 'entailment'
        # Else output class 'neutral'
        else:
            return 'neutral'


def estimate_z_and_log_lik(clf, y_hat, data, not_nan, n_samples):
    n = data.shape[0]
    preds = np.empty((y_hat.shape[0] * n_samples, len(use_z_values)), dtype=np.dtype('U25'))
    preds[:,:] = np.nan
    probs = np.zeros((n, len(use_z_values), 3))
    for i in range(len(use_z_values)):
        probs[not_nan[:,i],i,:] = clf[i].predict_proba(data[not_nan[:,i], :][:, cols[i]])

    def generate_z_samples(k):
        if k % 20000 == 0 and k != 0:
            print(f"{round(k / y_hat.shape[0] * 100, 2)}% of Z samples generated")
        n_samples_missing = n_samples
        res = np.empty((n_samples, len(use_z_values)), dtype=np.dtype('U25'))
        res[:,:] = np.nan
        temp = np.empty((0, len(use_z_values)), dtype=np.dtype('U25'))
        while n_samples_missing > 0:
            Foo = np.empty((n_samples_missing*10, len(use_z_values)), dtype=np.dtype('U25'))
            Foo[:,:] = np.nan
            for i in range(len(use_z_values)):
                if not_nan[k,i]:
                    Foo[:,i] = np.random.choice(clf[i].classes_, n_samples_missing*10, p=probs[k,i,:]).tolist()
            use_samples = (predict_y_from_z(Foo) == y_hat[k])
            temp = np.vstack((temp, Foo[use_samples,:]))
            n_samples_missing = n_samples - temp.shape[0]
        res = temp[:n_samples,:]
        return res

    preds = np.vstack(tuple([generate_z_samples(j) for j in range(y_hat.shape[0])]))

    # Train accuracy
    z_train_temp = np.empty((y_hat.shape[0], len(use_z_values)), dtype=np.dtype('U25'))
    z_train_temp[:,:] = np.nan
    for i in range(len(use_z_values)):
        z_train_temp[not_nan[:,i], i] = clf[i].predict(data[not_nan[:,i],:][:, cols[i]])
    predicted_y_train_temp = predict_y_from_z(z_train_temp)
    print(f"Current Train Accuracy Before Adaptions: {np.mean(predicted_y_train_temp == y_hat)}")

    # Calculate log-likelihood
    y_mapping = {clf[0].classes_[0]: 0, clf[0].classes_[1]: 1, clf[0].classes_[2]: 2}
    log_prob = np.zeros((len(use_z_values), 3, n))
    for i in range(len(use_z_values)):
        log_prob[i, :, not_nan[:,i]] = clf[i].predict_log_proba(data[not_nan[:,i],:][:, cols[i]])
    log_lik = np.sum([log_prob[:, y_mapping[predicted_y_train_temp[i]], i] for i in range(n)])
    print(log_lik)

    return preds, log_lik


def em_algorithm(clf, y_hat, data, not_nan, iter, dev_data, y_dev, not_nan_dev, n_samples):
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
        z, cur_log_lik = estimate_z_and_log_lik(clf, y_hat, data, not_nan, n_samples)
        log_lik += [cur_log_lik, ]
        print(f'Current log likelihood: {cur_log_lik}')

        # M-Step
        print(datetime.datetime.now())
        print("M-Step")
        models_not_updated = list()
        for i in range(len(use_z_values)):
            try:
                clf[i].fit(np.repeat(data[not_nan[:,i],][:,cols[i]], n_samples, axis=0), z[np.repeat(not_nan[:,i], n_samples, axis=0), i])
            except:
                models_not_updated += [i,]
            z_dev[not_nan_dev[:,i],i] = clf[i].predict(dev_data[not_nan_dev[:,i],][:,cols[i]])
        if len(models_not_updated) > 0:
            print(f"The following models are not updated because target y does not contain all classes: {models_not_updated}")
        y_pred_dev = predict_y_from_z(z_dev)
        y_pred_train = predict_y_from_z(z)
        print(f"Current Train Accuracy After Adaptions (should be 1): {np.mean(np.repeat(y_hat, n_samples, axis=0) == y_pred_train)}")
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

    n_samples = 8 # for training 300, sample = 12. For training 60, sample = 60. For training 480, sample = 8.
    continue_training = True
    prev_iter = 20
    amount_training_data = 480 # min 40, max 480
    batch_size = amount_training_data * 200
    em_iter = 5
    mlp_iter = 10
    size_hidden_layers = (200, 50, 30)
    str_size_hidden = '_'.join([str(i) for i in size_hidden_layers])
    if not continue_training:
        prev_iter = 0

    prev_directory = 'vers31/MLP_Classifiers_' + str(amount_training_data) + 'k_training_' + str(prev_iter) + '_iter_' + 'NN_size_' + str_size_hidden
    folder_name = 'vers31/MLP_Classifiers_' + str(amount_training_data) + 'k_training_' + str(prev_iter + em_iter) + '_iter_' + 'NN_size_' + str_size_hidden
    use_z_values = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)
    # Indices in terms of z for all hidden variables that are not mixed, e.g. Subject1-Subject2, Verb1-Verb2, etc.
    non_mixed_pairs_indices = [i for i in range(len(use_z_values)) if use_z_values[i] in (0,6,12,18,24)]

    # Indices for all z variables influenced by Subject2 (Verb2, Object2 etc. respectively) (e.g. Subject1-Subject2)
    Subj2_indices = [i for i in range(len(use_z_values)) if use_z_values[i] in (0, 5, 10, 15, 20)]
    Verb2_indices = [i for i in range(len(use_z_values)) if use_z_values[i] in (1, 6, 11, 16, 21)]
    Obj2_indices = [i for i in range(len(use_z_values)) if use_z_values[i] in (2, 7, 12, 17, 22)]
    Loc2_indices = [i for i in range(len(use_z_values)) if use_z_values[i] in (3, 8, 13, 18, 23)]
    Clo2_indices = [i for i in range(len(use_z_values)) if use_z_values[i] in (4, 9, 14, 19, 24)]
    Sentence2_indices = [Subj2_indices, Verb2_indices, Obj2_indices, Loc2_indices, Clo2_indices]

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

    # Load previous MLP Classifiers
    if continue_training:
        print(datetime.datetime.now())
        print("Load previous MLP Classifiers")
        clf = list()
        for i in range(len(use_z_values)):
            with open(prev_directory + "/MLP_Classifier" + str(i) + ".pkl", "rb") as f:
                clf += [pickle.load(f), ]
    # Initialise MLP Classifier
    else:
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

    # Store relevant y_hat values
    y_hat = list()
    for ind in data_index:
        label = original_dataset.loc[ind].gold_label
        y_hat += [label, ]
    y_hat = np.array(y_hat)

    # Initialise z
    if not continue_training:
        print(datetime.datetime.now())
        print("Initialise z")
        z_tuple = tuple(map(initialise_z, range(n_samples)))
        z = np.empty((y_hat.shape[0] * n_samples, len(use_z_values)), dtype=np.dtype('U25'))
        z[:,:] = np.nan
        for i in range(n_samples):
            z[np.arange(0,y_hat.shape[0] * n_samples, n_samples)+i,:] = z_tuple[i]
        print(f"Accuracy of initial z values (should be 1): {np.mean(predict_y_from_z(z) == np.repeat(y_hat, n_samples, axis=0))}")

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
    if not continue_training:
        print(datetime.datetime.now())
        print("Initial Training of MLP classifiers with initial z values")
        z_dev = np.empty((dev_data.shape[0], len(use_z_values)), dtype=np.dtype('U25'))
        z_dev[:,:] = np.nan
        for i in range(len(use_z_values)):
            clf[i].fit(np.repeat(data[not_nan[:,i],][:,cols[i]], n_samples, axis=0), z[np.repeat(not_nan[:,i], n_samples, axis=0), i])
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
                                            not_nan_dev=not_nan_dev,
                                            n_samples=n_samples)

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