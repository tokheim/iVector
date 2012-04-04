import os
import math

languageMapper = {'ENG':0, 'SPA':1, 'MAN':2, 'FAR':3, 'FRE':4, 'GER':5, 'HIN':6, 'JAP':7, 'KOR':8, 'TAM':9, 'VIE':10, 'ARA':11, 'OutOfSet':12}
callLanguages = [ 'ARABIC_EGYPT', 'ENG_GENRL', 'ENG_SOUTH', 'FARSI', 'FRENCH_CAN', 'GERMAN', 'HINDI', 'JAPANESE', 'KOREAN', 'MANDARIN_M', 'MANDARIN_T', 'SPANISH', 'SPANISH_CAR', 'TAMIL', 'VIETNAMESE' ]
callSets = ['train', 'devtest']
nistKeyFile = '/talebase/data/speech_raw/NIST_LR/2003/docs/LID03_KEY.v3'
nistindir = './NIST/2003/lid03e1/transcripts/30/'
unigramlistin = './other/unigramList.txt'
inf = 999999999999
#unigramlistin = './other/fullUnigramList.txt'

docs = [[],[],[]]#List for train, dev and evl sets
trainmodels = []#List of models (uni bi tri), same order as callLanguages

allUnigrams = {}
numberOfUnigrams = 0
kvalues = range(3, 30)


class DocInfo:
    def __init__(self, fpath, lang):
        self.lang = lang
        self.fpath = fpath

class LangModel:
    def __init__(self, probs, alphas):
        self.probs = probs
        self.alphas = alphas
    
    def getLikelihood(self, ngram):
        backoff = 0
        for i in range(len(ngram),0, -1):
            key = tuple(ngram[:i])
            histkey = tuple(ngram[1:i])
            if self.probs.has_key(key):
                return backoff + self.probs[key]
            elif i == 1:#Unigram
                return backoff -inf
            elif self.alphas.has_key(histkey):
                backoff += self.aplhas[histkey]

def getLabelNum(lang):
    str = lang[:3]
    if languageMapper.has_key(str):
        return languageMapper[str]
    else:
        return languageMapper['OutOfSet']



def isSilence(symbol):
  return allUnigrams['pau'] == allUnigrams[symbol] or allUnigrams['pau'] == symbol 

def isNoise(symbol):
  return symbol == 'spk' or symbol == 'int'

def updateNgrams(phone, phonelist):
    if isNoise(phone) or (isSilence(phone) and len(phonelist) > 0 and isSilence(phonelist[0])):
        return 0
    phonelist.insert(0, allUnigrams[phone])
    if len(phonelist)>3:
        phonelist.pop()
    return 1


def createLanguageModel(k, ngramcounts, totPhonemes):
    ngramprobs = {}
    for key, value in ngramcounts.items():
        if len(key) == 1:#Use unigram probabilities directly
            ngramprobs[key] = math.log(value)-math.log(totPhonemes)
    
    for i in range(2, 4):#Build probabilities recursively
        for key, value in ngramcounts.items():
            if len(key) == i and value > k:#ignore if less than k occurences
                historyOfGram = key[1:]
                ngramprobs[key] = math.log(value)-math.log(ngramcounts[historyOfGram])
    
    #Find alpha values for backoff smoothing
    alphaNominators = {}
    alphaDenominators = {}
    for i in range(2, 4):
        for key, value in ngramprobs.items():
            if len(key) == i:
                historyOfGram = key[1:]
                lessGram = key[:(i-1)]
                if alphanominators.has_key(historyOfGram):
                    alphanominators[historyOfGram] -= math.exp(value)
                    alphadenominators[historyOfGram] -= math.exp(ngramprobs[lessGram])
                else:
                    alphanominators[historyOfGram] = 1 - math.exp(value)
                    alphadenominators[historyOfGram] = 1 - math.exp(ngramprobs[lessGram])
    alphas = {}
    for key, value in alphaNominators.items():
        if alphadenominators[key] > 0 and value > 0:#Fyll in forklaring her
            alpha[key] = math.log(value/alphadenominators[key])
        elif alphanominators[key] == 0:#Nominator 0 means that none of the probability mass is used for smoothing
            alpha[key] = -inf
    
    return LangModel(ngramprobs, alphas)



#Calculates the likelihood of the document for a given list of models, returns array of likelihoods in same order
def calcLikelihoods(doc, models):
    likes = [0.0]*len(models)
    ngram = []
    inFile = os.open(doc.fpath, 'r')
    for line in inFile:
        splitLine = line.split()
        if len(splitLine) > 3:
            phone = splitLine[2]
            if updateNgrams(phone, ngram):
                for i in range(len(models)):
                    likes[i] += models[i].getLikelihood(ngram)
    inFile.close()
    return likes


#Calculate the likelihood for given document with given model
def calcLikelihood(doc, model):
    modelList = [].append(model)
    return calcLikelihoods(doc, modelList)[0]

#Calculates the likelihoods for each document and each model
def recognize(docs, models):
    return [calcLikelihoods(doc, models) for doc in docs]

#Saves probabilitie outputs including the correct label
def saveProbs(docs, results, savePath):
    outFile = os.open(savePath, 'w')
    for i in range(len(docs)):
        str = str(getLabelNum(docs[i].lang)+1)
        for val in results[i]:
            str+=' '+str(val)
        outFile.write(str+'\n')
    outFile.close()

#prints identification results
def printResult(docs, results, languageMapper, languages):
    tot = 0.0
    confMatrix = [[0 for _ in range(len(languageMapper))] for _ in range(len(languageMapper))]
    for i in range(len(docs)):
        bestIndex = 0
        for j in range(1, len(languages)):
            if results[i][j] > results[i][bestIndex]:
                bestIndex = j
        confMatrix[getLabelNum(docs[i].lang)][getLabelNum(languages[bestIndex])] += 1
        tot += 1
    correct = 0.0
    for i in range(len(confMatrix)):
        print str(confMatrix[i])
        correct += confMatrix[i][i]
    print str(correct/tot)
    

#Read document languages and positions into lists      
for i in range(len(callSets)):
    for language in callLanguages:
        path = './CallFriend/'+language+'/vectsplit/'+callSets[i]'/'
        for filename in os.listdir(path):
             if filename.endswith('.txt'):
                 docs[i].append(DocInfo(path+filename, language))
keyFile = os.open(nistKeyFile, 'r'):
for line in keyFile:
    splitLine = line.split()
    if os.path.isfile(nistindir+splitline[0]+'.rec'):
        docs[2].append(DocInfo(nistindir+splitline[0]+'.rec', splitline[1]))
keyFile.close()

#Read list of possible unigrams
unigramfile = open(unigramlistin, 'r')
for line in unigramfile:
    splitline = line.split(' ')
    for splits in splitline:
        allunigrams[splits.rstrip()] = numofunigrams
    numofunigrams += 1
unigramfile.close()

print 'Preparations done'
        
#Read n-gram probabilities for train set
for language in callLanguages:
    totPhonemes = 0.0
    ngramcounts = {}
    for doc in docs[0]:
        if doc.lang == language:
            ngrams = []#Holds upto the last three phones, newest first
            inFile = os.open(doc.fpath, 'r')
            for line in inFile:
                splitLine = line.split()
                if len(splitLine > 3):
                    phone = splitLine[2]
                    if updateNgrams(phone, ngrams):
                        totPhonemes += 1.0
                        
                    for j in range(len(ngrams)):
                        key = tuple(ngrams[:(j+1)])
                        if ngramcounts[key]:
                            ngramcounts[key]+=1.0
                        else:
                            ngramcounts[key]=1.0
                    
            inFile.close()
    
    print 'Probabilities for '+language+' read'
    
    #ngrams with count less than k is discarded, check which model gives best likelihood
    bestK = -1
    bestLike = -inf
    for kvalue in kvalues:
        model = createLanguageModel(kvalue, ngramcounts, totPhonemes)
        devlike = 0
        for doc in docs[1]:
            if doc.lang == callLanguages[i]:
                devlike += calcLikelihood(doc, model)
        if devLike > bestLike:
            bestK = kvalue
            bestLike = devLike
            print 'New best k '+str(kvalue)+' for '+language+' found, like: '+str(devLike)
        
    trainmodels.append(createLanguageModel(bestK, ngramcounts, totPhonemes))
    print '---Model for '+language+' created---'
    
print 'Models created, dev testing starting'
results = recognize(docs[1], trainmodels)
saveProbs(docs[1], results, 'dev_results.txt')
printResult(docs[1], results, languageMapper, callLanguages)

print 'Evl testing starting'
results = recognize(docs[2], trainmodels)
saveProbs(docs[2], results, 'evl_results.txt')
printResult(docs[2], results, languageMapper, callLanguages)
