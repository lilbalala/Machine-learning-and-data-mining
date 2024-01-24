import random

from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy

def gender_fearures(word, num_letters=2):
    return {'feature':word[-num_letters:].lower()}

if __name__ == '__main__':
    file_male = open('male.txt')
    file_female = open('female.txt')

    labeled_names = ([(name,'male') for name in file_male] +
                     [(name,'female') for name in file_female])

    random.seed(7)
    random.shuffle(labeled_names)
    input_names = ['Leonardo','Amy','Sam']

    for i in range(1,5):
        print('\nNumber of letters:',i)
        features = [(gender_fearures(n,i),gender) for (n,gender) in labeled_names]
        train_set, test_set = features[500:],features[:500]
        classifier = NaiveBayesClassifier.train(train_set)
        print('Accuracy ==>',str(100 * nltk_accuracy(classifier,test_set))+str('%'))

        for name in input_names:
            print(name,'==>',classifier.classify(gender_fearures(name,i)))
