from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math
class Domain:
    def __init__(self, _name,_length,_numbers,_segment,_vowel,_commonRoot,_entropy,_label):
        self.name=_name
        self.length=_length
        self.numbers=_numbers
        self.segment=_segment
        self.label=_label
        self.vowel = _vowel
        self.commonRoot=_commonRoot
        self.entropy=_entropy

    def returnTuple(self):
        return [self.length,self.numbers,self.segment,self.vowel,self.commonRoot,self.entropy]

    def returnLabel(self):
        if self.label=="dga":
            return 1
        else:
            return 0

def getRoot(str):
    commonRootList = ['cn', 'com', 'net', 'org', 'gov', 'info', 'edu']

    if str.split('.')[-1] in commonRootList:
        return 1
    else:
        return 0

def getNumbers(str):
    count=0
    for ch in str:
        if ch.isdigit():
            count+=1
    return count

def getVowel(str):
    VowelList = ['a','e', 'i', 'o', 'u']
    str = str.lower()
    countWord = 0
    countVowel = 0
    for ch in str:
        if ch.isalpha():
            countWord+=1
            if ch in VowelList:
                countVowel += 1
    if countWord==0:
        return 0
    else:
        return countVowel / countWord


def getSegment(str):
    count=0
    for ch in str:
        if ch == '.':
            count+=1
    return count

def getEntropy(str):
    sum = 0
    entropy = 0
    letter = [0] * 26
    str = str.lower()
    for i in range(len(str)):
        if str[i].isalpha():
            letter[ord(str[i]) - ord('a')] += 1
            sum += 1
    for i in range(26):
        p = letter[i] / sum
        if p > 0:
            entropy += -(p * math.log(p, 2))
    return entropy
if __name__ == '__main__':
    file = open('train.txt')
    trainList = []
    for line in file:
        line=line.strip()
        elements = line.split(',')
        name = elements[0]
        label = elements[1]
        trainList.append(Domain(name,len(name),getNumbers(name),getSegment(name),getVowel(name),getRoot(name),getEntropy(name),label))
    file.close()
    featureMatrix = []
    labelList = []
    for item in trainList:
        featureMatrix.append(item.returnTuple())
        labelList.append(item.returnLabel())
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    file = open('test.txt')
    testList = []
    for line in file:
        line = line.strip()
        name = line
        testList.append(Domain(name,len(name),getNumbers(name),getSegment(name),getVowel(name),getRoot(name),getEntropy(name),'label'))
    file.close()
    file = open('result.txt','w')
    for item in testList:
        file.write(item.name+",")
        if clf.predict([item.returnTuple()])==0:
            file.write("notdga\n")
        else:
            file.write("dga\n")
    file.close()

