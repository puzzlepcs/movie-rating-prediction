import pandas as pd
from konlpy import jvm
from konlpy.tag import Okt
import json
import nltk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import os
# %matplotlib inline

class KerasClassifier():
    def __init__(self, trainfile, testfile, size=10000, verbose=False):
        if os.path.isfile(trainfile):
            print('json file found. loading...')
            with open(trainfile,'r') as f:
                self.train_docs = json.load(f)
            with open(testfile, 'r') as f:
                self.test_docs = json.load(f)
            print('done.')
            
        else:
            train_data = self.readfile('data/ratings_train.txt')
            test_data = self.readfile('data/ratings_test.txt') 
            self.train_docs = [(self.tokenize(row[1]), row[2]) for row in train_data]
            self.test_docs = [(self.tokenize(row[1]), row[2]) for row in test_data]
            with open(trainfile, 'w', encoding='utf-8') as makefile:
                json.dump(train_data, makefile, ensure_ascii=False, indent='\t')
            with open(testfile, 'w', encoding='utf-8') as makefile:
                json.dump(test_data, makefile, ensure_ascii=False, indent='\t')
        
        print('tokenizing the dataset...')
        tokens = [t for d in self.train_docs for t in d[0]]
        text = nltk.Text(tokens, name='NMSC')
        self.selected_words = [f[0] for f in text.vocab().most_common(size)]

        if verbose:
            font_fname = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
            font_name = font_manager.FontProperties(fname = font_fname).get_name()
            rc('font', family=font_name)

            plt.figure(figsize=(20,10))
            text.plot(50)
        print('done.')

        self.x_train = None
        self.y_train= None
        self.x_test = None
        self.y_test = None
        self.size = size

    def get_vector(self):
        train_x = [self.term_frequency(d) for d,_ in self.train_docs]
        test_x = [self.term_frequency(d) for d,_ in self.test_docs]
        train_y = [c for _,c in self.train_docs]
        test_y = [c for _,c in self.test_docs]

        x_train = np.asarray(train_x, dtype='float32')
        x_test = np.asarray(test_x, dtype='float32')

        y_train = np.asarray(train_y, dtype='float32')
        y_test = np.asarray(test_y, dtype='float32')

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        return x_train, y_train, x_test, y_test

    def term_frequency(self, doc):        
        return [doc.count(word) for word in self.selected_words]
    
    def readfile(self,filename):
        f = open(filename, 'r')
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
        f.close()
        return data

    def tokenize(self,doc):
        okt = Okt()
        return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]
    
    def train(self, learning_rate):
        from tensorflow.keras import models
        from tensorflow.keras import layers
        from tensorflow.keras import optimizers
        from tensorflow.keras import losses
        from tensorflow.keras import metrics

        self.model = models.Sequential()
        self.model.add(layers.Dense(64, activation='relu', input_shape=(self.size, )))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        print('training start...')
        self.model.compile(optimizer=optimizers.RMSprop(lr=learning_rate),
                    loss=losses.binary_crossentropy,
                    metrics=[metrics.binary_accuracy]
        )
        print('done.')
    
    def fit(self, epoch, batchsize):
        self.model.fit(self.x_train, self.y_train, epochs=epoch, batch_size=batchsize)
    
    def evaluate(self):
        return self.model.evaluate(self.x_test, self.y_test)

    def predict_pos_neg(self,review):
        token = self.tokenize(review)
        tf = self.term_frequency(token)
        data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
        score = float(self.model.predict(data))

        if score > 0.5:
            print('[{}]: considered 1 (positive), {:.2f}% \n'.format(review, score*100))
        else:
            print('[{}]: considered 0 (negative), {:.2f}% \n'.format(review, score*100))

def main():
    classifyer = KerasClassifier('data/train_docs.json', 'data/test_docs.json', size=5000, verbose=False)
    x_train, y_train, x_test, y_test = classifyer.get_vector()
    classifyer.train(0.001)
    classifyer.fit(10, 512)
    results = classifyer.evaluate()
    print(results)

    print('\n')
    classifyer.predict_pos_neg('올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요.')
    classifyer.predict_pos_neg('배경 음악이 영화의 분위기랑 너무 안 맞앗습니다. 몰입에 방해가 됩니다.')
    classifyer.predict_pos_neg('주연 배우가 신인인데 연기를 진짜 잘 하네요. 몰입감 ㅎㄷㄷ')
    classifyer.predict_pos_neg('주연배우 때문에 봤어요')
    classifyer.predict_pos_neg('진짜 너무 너무')
    

if __name__ == "__main__":
    jvm.init_jvm()
    main()