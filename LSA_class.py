from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition.truncated_svd import TruncatedSVD
import re
import unicodedata
from sklearn.utils.extmath import randomized_svd
import numpy as np
from numpy.linalg import inv
from scipy.sparse import csr_matrix
from nltk.stem import *
from nltk import word_tokenize
import nltk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import time
import constants
import pickle
import sys
import os

class LSA_pipeline:


    def __init__(self):
        #change your csv name !!
        # You have to provide your csv name
        # Tabs
        self.csv_name = "temp" 

        #For cleaning
        self.stemmer = nltk.stem.SnowballStemmer('english')
        self.regex_re = re.compile('[^a-z0-9!,.?\'"-:; ]')

        #LSA Raw_Text
        self.raw_text = "This is a sample raw text, this text will be processed and necessary pickles will be created you can give this input from terminal or somewhere else "

        #LSA number of latent factors
        self.number_of_factors = 200

        # Necessary matrix variables for Latent semantic analysis
        self.x = ""
        self.u = ""
        self.sigma = ""
        self.vt = ""
        self.q = ""
        self.raw_latent_factor = ""
        self.latent_factors = ""
        self.q_truncated = ""
        self.x_truncated = ""

        ##csv variables
        self.clean = []
        self.name = []
        self.tokenized = []
        self.df = ""

        ##Tf-idf settings
        self.vocabulary = ""
        self.max_df = 0.85
        self.min_df = 2
        self.vectorizer2 = ""

        ##Elasticsearch clients
        self.ES_host = "temp" # You Elastic Search
        #self.es = Elasticsearch([{'host': self.ES_host}])
        #self.es.info(pretty=True)

        ##Org_id - You have to provide index and type
        self.index = "temp" # elastic search index and type
        self.type = "temp"

        ##create pickle directory for pickles
        self.directory = os.getcwd() + '/pickles'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)


    #text normalizer
    def py_asciify(self, text):
        return unicodedata.normalize('NFKD', text.replace(u'\u0131', 'i').replace(u"\u00A5",'y').replace(u"\u2122",' ').replace(u"\u00E2",'a').replace(u')', '\)').replace(u'(', '\(')).encode('ascii', 'ignore').decode('utf8').lower()

    def cleaner(self, text):
        text = py_asciify(text)
        tokenized = nltk.word_tokenize(text)
        regex = re.compile('[^a-z0-9!,.?\'"-:; ]')
        text = regex.sub('', text)
        if text == "nan" or text == "" or text == None:
            text = "empty"

        tokenized = nltk.word_tokenize(text)
        total = ""
        for i in tokenized:
            total+= self.stemmer.stem(i) + " "
        return total

    def prepare_data(self):
        self.df = pd.read_csv(self.csv_name, header=[0], sep= '\t')

        if not 'tokenized' in self.df:
            tokenized = []
            for row in self.df['clean']:
                tokenized.append( self.cleaner(row) )
            self.df['tokenized'] = tokenized
            self.df.to_csv(self.csv_name, index=False, sep = '\t')

        self.clean = []
        for i in self.df['tokenized']:
            if str(i) == "nan":
                i = ""
            self.clean.append(str(i))
        print(" -- prepare_data done")

    def Tf_idf(self):
        vectorizer = TfidfVectorizer(stop_words = "english", analyzer = 'word', max_df = self.max_df)
        self.x = vectorizer.fit_transform(self.clean)
        self.vocabulary = vectorizer.vocabulary_
        pickle.dump(self.x, open( (self.directory + "/X.p") , "wb" ))
        pickle.dump(self.vocabulary, open((self.directory + "/vocabulary.p"), "wb" ))
        print("-- TF_IDF is done")

    def raw_text_Tf(self):
        self.vectorizer2 = TfidfVectorizer(stop_words = "english", analyzer = 'word', vocabulary=self.vocabulary, max_df = self.max_df, min_df= self.min_df)
        not_important = self.vectorizer2.fit_transform(self.clean)
        pickle.dump(self.vectorizer2, open( (self.directory + "/vectorizer2.p" ), "wb" ))
        self.q = self.vectorizer2.fit_transform([self.raw_text])
        print("-- raw_text_Tf done")

    def LSA(self):
        self.x = self.x.transpose()
        self.u, self.sigma, self.vt = randomized_svd(self.x,
                                      n_components=self.number_of_factors,
                                      n_iter=5,
                                      random_state=None)
        #creating pickles
        pickle.dump(self.u, open( (self.directory + "/U.p"), "wb" ))
        pickle.dump(self.sigma, open( (self.directory + "/Sigma.p"), "wb" ))
        pickle.dump(self.vt, open( (self.directory + "/VT.p") , "wb" ))
        print("-- LSA done")

    def LSA_help_1(self):
        self.sigma = np.eye(self.number_of_factors)*self.sigma
        inv_sigma = inv(self.sigma)
        U_transpose = self.u.transpose()
        self.x_truncated = inv_sigma.dot(U_transpose)
        self.x_truncated = csr_matrix(self.x_truncated)
        self.q_truncated = self.x_truncated
        #creating pickles
        pickle.dump(self.x_truncated, open( (self.directory + "/x_truncated.p"), "wb" ))
        print("-- LSA_help_1 done")

    def LSA_help_update(self):
        xx = csr_matrix(self.x)
        self.x_truncated = self.x_truncated.dot(xx)
        self.latent_factors = self.x_truncated.transpose()
        print(" -- LSA_help_update done")

    def LSA_help_Raw_text(self):
        qq = csr_matrix(self.q)
        qq = qq.transpose()
        self.q_truncated = self.q_truncated.dot(qq)
        self.raw_latent_factor = self.q_truncated.transpose()
        print("-- LSA_help_Raw_text done")

    def save_latents(self):
        update_list = []

        for latents in self.latent_factors:
            temp = {}
            temp['latent'] = latents
            update_list.append(temp)

        for i in update_list:
            latt = i['latent'].toarray()[0]
            latent_fac = ""
            for count,j in enumerate(latt,0):
                latent_fac+= str(count) + "|" + str(j.item()) + " "
            latent_fac = latent_fac[0:-1]
            i['latent_final'] = latent_fac

        latents_list = []
        for i in update_list:
            latents_list.append(i["latent_final"])

        self.df['latents'] = latents_list
        self.df.to_csv(self.csv_name, index=False, sep = '\t')
        print("-- save_latents done")

    def query(self, factors):
        q = {
            "script" : {
                "inline": "ctx._source.lsa_factors = params.factors",
                "lang": "painless",
                "params" : {
                    "factors" : factors
                }
            }
        }
        return q

    def update(self, ids, LSA_factors):
        for id,factors in zip(self.df['assetId'],self.df['latents']):
            q = query(factors)
            try:
                es.update(body=q, index=self.index, doc_type=self.type, id = id)
            except:
                print(count)
        print("--update done")

    def run_all(self,org_id,personal_id):
        print("--run_all started")
        self.prepare_data()
        self.Tf_idf()
        self.raw_text_Tf()
        self.LSA()
        self.LSA_help_1()
        self.LSA_help_update()
        self.LSA_help_Raw_text()
        self.save_latents()
        print("--run_all ended")
        ## After this operation prepare your Elasticsearch indexes and run update()
        ## self.update()

## User have to provide org_id and personal_id when running the script
if __name__ == "__main__":
    org_id = sys.argv[1]
    personal_id = sys.argv[2]
    LSA_pipeline().run_all(org_id=org_id, personal_id=personal_id)
