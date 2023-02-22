#%%
import textacy
import pandas as pd
import spacy
from subject_verb_object_extract import findSVOs
import datetime
import numpy as np

nlp = spacy.load('en_core_web_lg')

#%%
def extract_SVO(text):
    tuples_list = list()
    tuples = textacy.extract.subject_verb_object_triples(text)
    if tuples:
        tuples_to_list = list(tuples)
        tuples_list.append(tuples_to_list)
    return tuples_list
#%%
def findLoc(text):
    for token in text:
        if token.dep_ == 'prep' and token.lemma_ in ['on', 'in']:
            return list(token.subtree)
#%%
def transform_svo_to_nlp(svos):
    res = list()
    for svo in svos:
        res.append([[token for token in nlp(sub_part)] for sub_part in svo])
    return res

#%%
def cosine_sim(v1, v2):
    return np.dot(v1, v2)  / (np.linalg.norm(v1) * np.linalg.norm(v2))

#%%
clothing_words = ['jeans', 'shirt', 'shirts', 'boot', 'boots', 'jacket', 'jackets', 'glove', 'gloves', 'shoe', 'shoes', 'sweater', 'sweaters', 'hat', 'hats', 'skirt', 'skirts', 'beanie', 'beanies', 'blouse', 'blouses', 'tank top', 'tank tops', 'shorts', 'trousers', 'pants', 'skirt', 'skirts', 'jumper', 'suit', 'suits', 'uniform', 'uniforms', 'dress', 'dresses', 'coat', 'coats', 'pullover', 'pullovers', 'sweatshirt', 'sweatshirts', 'cardigan', 'cardigans', 'sandle', 'sandles', 'raincoat', 'raincoats', 'swimsuit', 'scarf', 'hoodie']

clothing_vectors = [nlp.vocab[clothing_word].vector for clothing_word in clothing_words]

def findClothing(text):
    # https://stackoverflow.com/questions/53493052/any-elegant-solution-for-finding-compound-noun-adjective-pairs-from-sentence-by
    for token in text:
        if any(np.array([cosine_sim(token.vector, v2) for v2 in clothing_vectors]) > 0.7):
            comps = [j for j in token.children if j.pos_ in ['ADJ', 'NOUN', 'PROPN']]
            if len(comps) > 0:
                return comps + [token, ]
            else:
                return [token, ]


#%%
def remove_clo_from_svo(clo, svo):
    if clo is not None:
        clo = clo.copy()
        svo = svo.copy()
        clo_lemmas = [cur_clo.lemma_ for cur_clo in clo]
        for i in range(len(svo)):
            svo[i][0] = [token for token in svo[i][0] if token.lemma_ not in clo_lemmas]
    return svo

#%%
def get_vectors(svo_s1, svo_s2, loc_s1, loc_s2, clo_s1, clo_s2, pairID, we_dim=300):
    if len(svo_s1) >= 1 and len(svo_s2) >= 1:
        if loc_s1 is not None:
            loc_vec_s1 = list(sum([token.vector for token in loc_s1]))
        else:
            loc_vec_s1 = we_dim * [np.nan, ]

        if loc_s2 is not None:
            loc_vec_s2 = list(sum([token.vector for token in loc_s2]))
        else:
            loc_vec_s2 = we_dim * [np.nan, ]

        if clo_s1 is not None:
            clo_vec_s1 = list(sum([token.vector for token in clo_s1]))
        else:
            clo_vec_s1 = we_dim * [np.nan, ]

        if clo_s2 is not None:
            clo_vec_s2 = list(sum([token.vector for token in clo_s2]))
        else:
            clo_vec_s2 = we_dim * [np.nan, ]

        cur_svo_s1 = svo_s1[0]
        cur_svo_s2 = svo_s2[0]

        if len(cur_svo_s1[0]) > 0:
            subj_vec_s1 = list(sum([token.vector for token in cur_svo_s1[0]]))
        else:
            subj_vec_s1 = we_dim * [np.nan, ]
        if len(cur_svo_s2[0]) > 0:
            subj_vec_s2 = list(sum([token.vector for token in cur_svo_s2[0]]))
        else:
            subj_vec_s2 = we_dim * [np.nan, ]
        verb_vec_s1 = list(sum([token.vector for token in cur_svo_s1[1]]))
        verb_vec_s2 = list(sum([token.vector for token in cur_svo_s2[1]]))
        if len(cur_svo_s1) == 3:
            obj_vec_s1 = list(sum([token.vector for token in cur_svo_s1[2]]))
        else:
            obj_vec_s1 = we_dim * [np.nan, ]
        if len(cur_svo_s2) == 3:
            obj_vec_s2 = list(sum([token.vector for token in cur_svo_s2[2]]))
        else:
            obj_vec_s2 = we_dim * [np.nan, ]
        return np.array([pairID, ] + subj_vec_s1 + verb_vec_s1 + obj_vec_s1 + loc_vec_s1 + clo_vec_s1 +
                subj_vec_s2 + verb_vec_s2 + obj_vec_s2 + loc_vec_s2 + clo_vec_s2).reshape(-1)

def subj_to_string(svo):
    try:
        svo = svo[0][0]
        res = [token.text for token in svo]
        res = " ".join(res)
    except:
        res = ""
    return res

def verb_to_string(svo):
    try:
        svo = svo[0][1]
        res = [token.text for token in svo]
        res = " ".join(res)
    except:
        res = ""
    return res

def obj_to_string(svo):
    try:
        svo = svo[0][2]
        res = [token.text for token in svo]
        res = " ".join(res)
    except:
        res = ""
    return res

def doc_to_string(doc):
    try:
        res = [token.text for token in doc]
        res = " ".join(res)
    except:
        res = ""
    return res


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

    full_train_set = train
    for cur_counter in np.arange(20,500,20):
        #cur_counter = 460
        print(cur_counter)
        print(datetime.datetime.now())
        train = full_train_set.iloc[(cur_counter-20) * 1000 : cur_counter * 1000]

        #%%
        # Word Embedding Dimension
        we_dim = 300
        data_name = 'train'
        if data_name == 'dev':
            data = dev
        elif data_name == 'train':
            data = train
        print(datetime.datetime.now())
        print("find nlp sentence 1")
        data['nlp_s1'] = data['Sentence1'].apply(nlp)
        print("find nlp sentence 2")
        data['nlp_s2'] = data['Sentence2'].apply(nlp)

        print(datetime.datetime.now())
        print("find SVO sentence 1")
        data['svo_s1'] = data['nlp_s1'].apply(findSVOs).apply(transform_svo_to_nlp)
        data['string_subj_s1'] = data['svo_s1'].apply(subj_to_string)
        data['string_verb_s1'] = data['svo_s1'].apply(verb_to_string)
        data['string_obj_s1'] = data['svo_s1'].apply(obj_to_string)
        print("find SVO sentence 2")
        data['svo_s2'] = data['nlp_s2'].apply(findSVOs).apply(transform_svo_to_nlp)
        data['string_subj_s2'] = data['svo_s2'].apply(subj_to_string)
        data['string_verb_s2'] = data['svo_s2'].apply(verb_to_string)
        data['string_obj_s2'] = data['svo_s2'].apply(obj_to_string)

        print(datetime.datetime.now())
        print("find Loc sentence 1")
        data['loc_s1'] = data['nlp_s1'].apply(findLoc)
        data['string_loc_s1'] = data['loc_s1'].apply(doc_to_string)
        print("find Loc sentence 2")
        data['loc_s2'] = data['nlp_s2'].apply(findLoc)
        data['string_loc_s2'] = data['loc_s2'].apply(doc_to_string)

        print(datetime.datetime.now())
        print("find Clo sentence 1")
        data['clo_s1'] = data['nlp_s1'].apply(findClothing)
        data['string_clo_s1'] = data['clo_s1'].apply(doc_to_string)
        print("find Clo sentence 2")
        data['clo_s2'] = data['nlp_s2'].apply(findClothing)
        data['string_clo_s2'] = data['clo_s2'].apply(doc_to_string)

        data['svo_s1'] = data.apply(lambda x: remove_clo_from_svo(x.clo_s1, x.svo_s1), axis=1)
        data['svo_s2'] = data.apply(lambda x: remove_clo_from_svo(x.clo_s2, x.svo_s2), axis=1)

        print(datetime.datetime.now())
        print("get embedding vectors")
        data['vecs'] = data.apply(lambda x: get_vectors(svo_s1=x.svo_s1, svo_s2=x.svo_s2,
                                                          loc_s1=x.loc_s1, loc_s2=x.loc_s2,
                                                          clo_s1=x.clo_s1, clo_s2=x.clo_s2,
                                                          pairID=x.pairID), axis=1)

        prepared_data = np.array([vec for vec in data.vecs if vec is not None])
        print(prepared_data.shape)




        prepared_data = pd.DataFrame(prepared_data)
        prepared_data.to_csv('prepared_data/subphrase_vectors_' + str(cur_counter) + data_name + '.csv', sep=';')
        data[["pairID", "string_subj_s1", "string_verb_s1", "string_obj_s1", "string_loc_s1", "string_clo_s1",
              "string_subj_s2", "string_verb_s2", "string_obj_s2", "string_loc_s2", "string_clo_s2"]].to_csv('prepared_data/subphrases_' + str(cur_counter) + data_name + '.csv')
