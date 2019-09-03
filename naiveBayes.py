# Naive Baysian implementation
from konlpy.tag import Okt
import math

class NaiveBayesClassifier:
    def __init_(self):
        self.num0 = 0           # total num of negative reviews
        self.num1 = 0           # total num of posivie reviews
        self.words0 = 0         # total num of words in negative reviews  
        self.words1 = 0         # total num of words in positive reviews
        self.train_dict0 = {}   # negative dictionary
        self.train_dict1 = {}   # positive dictionary

    def update_dict(self, comment, label):
        # positive review
        if label=='1': 
            self.num1 += 1
            self.words1 += len(comment)
        # negative review
        elif label=='0':
            self.num0 += 1
            self.words0 += len(comment)

        for j in range(len(comment)):
            word = comment[j]
            if label == '1':
                if word in self.train_dict1:
                    self.train_dict1[word] += 1
                else:
                    self.train_dict1[word] = 1
            elif label == '0':
                if word in self.train_dict0:
                    self.train_dict0[word] += 1
                else:
                    self.train_dict0[word] = 1
    
    # train the model parsed by token
    def train_by_token(self, filename):
        f = open(filename, 'r')
        print('Train by token')
        print('Start training...')

        i = 0
        while True:
            i += 1
            if i % 10000 == 0:
                print('processing {}th line'.format(i))

            # parsing data
            line = f.readline()
            if not line: 
                break
            tmp = list(line.strip('\n').split('\t'))
            try:
                (_,c,l) = tmp
            except:
                continue
            comment = c.split()
            self.update_dict(comment, l)
        f.close()
    
    # train the model parsed by morph
    def train_by_morph(self, filename):
        twit = Okt()
        
        f = open(filename, 'r')
        print('Train by morph')
        print('Start training...')

        i = 0
        while True:
            i += 1
            if i % 10000 == 0:
                print('processing {}th line'.format(i))

            # parsing data
            line = f.readline()
            if not line: 
                break
            tmp = list(line.strip('\n').split('\t'))
            try:
                (_,c,l) = tmp
            except:
                continue

            comment = twit.morphs(c)
            self.update_dict(comment, l)
        f.close()

    def calculate_prob(self, comment):
        p1 = 0
        p0 = 0
        v = len(self.train_dict0.keys()) + len(self.train_dict1.keys())
        p1 += math.log(self.num1 / (self.num1 + self.num0))
        p0 += math.log(self.num0 / (self.num0 + self.num1))
        
        for j in range(len(comment)):
            word = comment[j]
            tmp1, tmp0 = 0, 0
            if word in self.train_dict1:
                tmp1 = self.train_dict1[word]
            if word in self.train_dict0:
                tmp0 = self.train_dict0[word]
            
            # Laplace smoothing
            p1 += math.log((tmp1+1)/(self.words1+v))
            p0 += math.log((tmp0+1)/(self.words0+v))
        
        return p1, p0

    def classify_by_token(self, input_file, output_file):
        fin = open(input_file, 'r')
        fin.readline()

        print('Start classifying...')

        ans = []
        ans.append('id\tdocument\tlabel')
        
        i = 0
        while True:
            i += 1
            if i % 10000 == 0:
                print('processing {}th line'.format(i))

            line = fin.readline()
            if not line: break

            tmp = list(line.strip('\n').split('\t'))
            try:
                (n, c) = tmp[:2]            # n: id num of the comment, c: unparsed comment
            except:
                continue

            comment = c.split()
            p1, p0 = self.calculate_prob(comment)
            if p1 > p0:
                l = '1'
            else:
                l = '0'
            ans_line ='\t'.join([n,c,l])
            ans.append(ans_line)

        fout = open(output_file, 'w') 
        fout.write('\n'.join(ans))
        fin.close()
        fout.close()

        print('Classification done')

    def classify_by_morph(self, input_file, output_file):
        twit = Okt()

        fin = open(input_file, 'r')
        fin.readline()

        print('Start classifying...')

        ans = []
        ans.append('id\tdocument\tlabel')
        
        i = 0
        while True:
            i += 1
            if i % 10000 == 0:
                print('processing {}th line'.format(i))

            line = fin.readline()
            if not line: break
    
            tmp = list(line.strip('\n').split('\t'))
            try:
                (n, c) = tmp[:2]            # n: id num of the comment, c: unparsed comment
            except:
                continue

            comment = twit.morphs(c)
            p1, p0 = self.calculate_prob(comment)
            if p1 > p0:
                l = '1'
            else:
                l = '0'
            ans_line ='\t'.join([n,c,l])
            ans.append(ans_line)

        fout = open(output_file, 'w') 
        fout.write('\n'.join(ans))
        fin.close()
        fout.close()

        print('Classification done')


if __name__=='__main__':
    train = 'data/ratings_train.txt'
    test = 'ratings_test.txt'
    result = 'ratings_result.txt'
    vaild = 'ratings_valid.txt'

    model = NaiveBayesClassifier()
    model.train_by_morph(train)
    model.classify_by_morph(test, result)