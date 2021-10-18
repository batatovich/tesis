import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import random as random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Labels: susceptible, incubating, infected, recovered.
#Problem statement: given a set of embeddings of a network, the goal is to label correctly all nodes taking as 
#an input the labels of a subset of nodes.

#susceptible = 0
#incubating  = 1
#infected    = 2
#recovered   = 3

class OneVsRest():
    def __init__(self, embeddings, states):
        self.embeddings = embeddings
        self.states = states

        L = len(self.states)
        ids = [i for i in range(L)]
        train_ids = random.sample(ids, int(L*0.7))
        test_ids = [i for i in ids if i not in train_ids]

        self.train_embeddings = [self.embeddings[i] for i in train_ids]
        self.test_embeddings = [self.embeddings[i] for i in test_ids]
        self.train_states = [self.states[i] for i in train_ids]
        self.test_states = [self.states[i] for i in test_ids]

        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.accuracy = 0
        self.F1 = 0
        self.presicion = 0
        self.recall = 0

    def train(self):
        self.clf = OneVsRestClassifier(SVC()).fit(self.train_embeddings, self.train_states) #How does it work?

    def test(self):
        test_prediction = self.clf.predict(self.test_embeddings)
        for i in range(len(self.test_states)):
            if test_prediction[i] == self.test_states[i]:
                self.accuracy += 1
                if test_prediction[i] == 2 or test_prediction[i] == 1: #If its infected or incubating
                    self.true_positives += 1
                else:
                    self.true_negatives += 1
            else:
                if test_prediction[i] == 2 or test_prediction[i] == 1 : #If its infected or incubating
                    self.false_positives += 1
                else:
                    self.false_negatives += 1   
        self.accuracy = self.accuracy/len(self.test_states)
        if self.true_positives != 0 and self.false_negatives != 0:
            self.recall = self.true_positives/(self.true_positives + self.false_negatives)
        if self.true_positives != 0 and self.false_positives != 0:
            self.presicion = self.true_positives/(self.true_positives + self.false_positives)
        if self.recall != 0 or self.presicion != 0:
            self.F1 = 2*self.presicion*self.recall/(self.presicion+self.recall)
        print('Overall accuracy: ' + str(self.accuracy) + '.' )
        print('Recall: ' + str(self.recall) + '.')
        print('Presicion: ' + str(self.presicion) + '.')
        print('F1 score: ' + str(self.F1) + '.')

    def predict(self, embedding):
        return self.clf.predict([embedding])



class NeuralNetwork():
    def __init__(self, embeddings, states, h_dim): #i_dim == embedding dimention, h_dim: number of features desired.
        self.states = states
        self.embeddings = embeddings

        L = len(self.states)
        ids = [i for i in range(L)]
        train_ids = random.sample(ids, int(L*0.7))
        test_ids = [i for i in ids if i not in train_ids]

        self.train_embeddings = np.array([self.embeddings[i] for i in train_ids])
        self.test_embeddings = np.array([self.embeddings[i] for i in test_ids])
        self.train_states = np.array([self.states[i] for i in train_ids])
        self.test_states = np.array([self.states[i] for i in test_ids])

        self.model = keras.Sequential([
        keras.Input(shape = (len(embeddings[0]),)),
        layers.Dense(h_dim, activation="relu", name="second_layer"),
        layers.Dense(4, activation = "softmax",name="output_layer")])
        self.model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    def train(self, ep):
        self.model.fit(self.train_embeddings, self.train_states, epochs = ep)
    def test(self):
        self.model.evaluate(
    x=self.test_embeddings, y=self.test_states, batch_size=None, verbose=1, sample_weight=None, steps=None,
    callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
    return_dict=False)
    def predict(self, emb):
        return self.model.predict(emb, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False)