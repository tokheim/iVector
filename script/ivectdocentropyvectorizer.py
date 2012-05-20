'''
Created on Jan 31, 2012

@author: Tokminator
'''
import os
import math
import copy

#Setup
languages = [ 'ARABIC_EGYPT', 'ENG_GENRL', 'ENG_SOUTH', 'FARSI', 'FRENCH_CAN', 'GERMAN', 'HINDI', 'JAPANESE', 'KOREAN', 'MANDARIN_M', 'MANDARIN_T', 'SPANISH', 'SPANISH_CAR', 'TAMIL', 'VIETNAMESE' ]
inbasedir = './CallFriend/?/vectsplit/'
nistindir = './NIST/2003/lid03e1/transcripts/30/'
setdir = ['train/', 'devtest/', 'shorttrain/']
outbasedir = './CallFriend/?/docnumvectors/'
unigramlistin = './other/unigramList.txt'
#unigramlistin = './other/fullUnigramList.txt'
outfilelistnames = ['./other/train_list.txt', './other/devtest_list.txt', './other/short_train_list.txt','./other/nist_list.txt']
nistoutdir = './NIST/2003/lid03e1/docnumvectors/30/'
nistkeyfile = '/talebase/data/speech_raw/NIST_LR/2003/docs/LID03_KEY.v3'
onlyTrigrams = 1

#Parameter initialization
allunigrams = {}
numofunigrams = 0

class DocInfo:
    def __init__(self, fname, lang):
        self.lang = lang
        self.fname = fname

def isSilence(symbol):
    return symbol == 'sil' or symbol == 'pau'

def isNoise(symbol):
    return symbol == 'spk' or symbol == 'int'

#To avoid feature ending with \n
def normalizeFeature(feature):
    return feature.rstrip()

def insert_feature_num(floatnum, vector):
    num = int(floatnum)
    if vector.has_key(num):
        vector[num] += 1
    else:
        vector[num] = 1

def insert_features(ngram, vector):
    lastnum = 0
    for i in range(len(ngram)):
        num = allunigrams[ngram[i]]*math.pow(numofunigrams, i)
        if i > 0:
            num += lastnum+math.pow(numofunigrams, i)
        insert_feature_num(num, vector)
        lastnum = num
            
def insert_feature(trigram, vector):
    if len(trigram) == 3:
        num = 0
        for i in range(3):
            num += allunigrams[trigram[i]]*math.pow(numofunigrams, 2-i)
        insert_feature_num(num, vector)

#Read file and create vector
def vectorize(indir, filename):
    tranfile = open(indir+filename, 'r')
    lastphones = []
    docvector = {}
    for line in tranfile:
        splitline = line.split(' ')
        if len(splitline) > 3:
            phone = normalizeFeature(splitline[2])
                    
            if isNoise(phone) or (isSilence(phone) and len(lastphones) > 0 and isSilence(lastphones[0])):
                continue
                    
            lastphones.insert(0, phone)
            if len(lastphones) > 3:
                lastphones.pop(3)
                
            if onlyTrigrams:
                insert_feature(lastphones, docvector)
            else:
                insert_features(lastphones, docvector)
    return docvector

#Read file and write vector
def makeVector(indir, outdir, filename, langEntropy, docEntropy):
    docvector = vectorize(indir, filename)
    outfile = open(outdir+filename.replace('.rec', '.txt'), 'w')
    for key, value in docvector.items():
        if langEntropy.has_key(key):
            outfile.write(str(key)+' '+str(value)+' '+str(math.sqrt(value))+' '+str(docEntropy[key]*value)+' '+str(langEntropy[key]*value)+'\n')        
    print 'Finished with file '+outdir+filename

def calcEntropy(key, tot, occurrencelist):
    entropy = 0.0
    for occurrencedict in occurrencelist:
        if occurrencedict.has_key(key):
            
            
            entropy += occurrencedict[key]*math.log(float(occurrencedict[key])/tot)
    return 1-entropy/(tot*math.log(len(occurrencelist)))

#Read list of possible unigrams
unigramfile = open(unigramlistin, 'r')
for line in unigramfile:
    splitline = line.split(' ')
    for splits in splitline:
        allunigrams[normalizeFeature(splits)] = numofunigrams
    numofunigrams += 1

#Find normalized entropy
langOccurrences = [{} for _ in range(len(languages))]
docOccurrences = []
allOccurrences = {}
for i in range(len(languages)):
    indir = inbasedir.replace('?', languages[i])+setdir[0]
    for filename in os.listdir(indir):
        docvector = vectorize(indir, filename)
        docOccurrences.append(copy.deepcopy(docvector))
        for key, value in docvector.items():
            if langOccurrences[i].has_key(key):
                langOccurrences[i][key]+=value
            else:
                langOccurrences[i][key]=value
            if allOccurrences.has_key(key):
                allOccurrences[key]+=value
            else:
                allOccurrences[key]=value
    print str((i+1)*100/len(languages))+'% finished reading files for entropy estimating'
langEntropy = {}
docEntropy = {}
for key, value in allOccurrences.items():
    langEntropy[key] = calcEntropy(key, value, langOccurrences)
    docEntropy[key] = calcEntropy(key, value, docOccurrences)
    
print 'Entropies calculated'


#Create training and devtest vectors
for i in range(len(setdir)):
    alldocs = []
    for language in languages:
        outdir = outbasedir.replace('?', language)+setdir[i]
        indir = inbasedir.replace('?', language)+setdir[i]
        os.system('mkdir -p '+outdir)
        os.system('rm '+outdir+'*.*')
        for filename in os.listdir(indir):
            makeVector(indir, outdir, filename, langEntropy, docEntropy)
            alldocs.append(DocInfo(outdir+filename, language))
    outfile = open(outfilelistnames[i], 'w')
    for doc in alldocs:
        outfile.write(doc.lang+' '+doc.fname+'\n')

#Create evaluation/NIST vectors
alldocs = []
keyfile = open(nistkeyfile, 'r')
os.system('mkdir -p '+nistoutdir)
os.system('rm '+nistoutdir+'*.*')
for line in keyfile:
    splitline = line.split()
    if not os.path.isfile(nistindir+splitline[0]+'.rec'):
        continue
    splitline = line.split(' ')
    makeVector(nistindir, nistoutdir, splitline[0]+'.rec', langEntropy, docEntropy)
    alldocs.append(DocInfo(nistoutdir+splitline[0]+'.txt', splitline[1]))
outfile = open(outfilelistnames[3], 'w')
for doc in alldocs:
    outfile.write(doc.lang+' '+doc.fname+'\n')
    

print 'Finished, total number of unigrams: '+str(numofunigrams)+', trigrams: '+str(math.pow(numofunigrams, 3))
if not onlyTrigrams:
    print 'Total: '+str(numofunigrams+math.pow(numofunigrams, 2)+math.pow(numofunigrams, 3))