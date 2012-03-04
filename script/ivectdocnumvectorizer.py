'''
Created on Jan 31, 2012

@author: Tokminator
'''
import os
import math

#Setup
languages = [ 'ARABIC_EGYPT', 'ENG_GENRL', 'ENG_SOUTH', 'FARSI', 'FRENCH_CAN', 'GERMAN', 'HINDI', 'JAPANESE', 'KOREAN', 'MANDARIN_M', 'MANDARIN_T', 'SPANISH', 'SPANISH_CAR', 'TAMIL', 'VIETNAMESE' ]
inbasedir = './CallFriend/?/vectsplit/'
nistindir = './NIST/2003/lid03e1/transcripts/30/'
setdir = ['train/', 'devtest/']
outbasedir = './CallFriend/?/docnumvectors/'
#NOT USED splitsymbol = '.'
unigramlistin = './other/unigramList.txt'
outfilelistnames = ['./other/train_list.txt', './other/devtest_list.txt', './other/nist_list.txt']
nistoutdir = './NIST/2003/lid03e1/docnumvectors/30/'
nistkeyfile = '/talebase/data/speech_raw/NIST_LR/2003/docs/LID03_KEY.v3'
#NOT USED - unigramlistout = './other/unigramlist.txt'

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

#1 to 1 maps feature strings to ints
def calc_feature_num(trigram):
    num = 0;
    for i in range(3):
        num += allunigrams[trigram[i]]*math.pow(numofunigrams, 2-i)
    return int(num)

#Read file and write vector
def vectorize(indir, outdir, filename):
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
                ngramnum = calc_feature_num(lastphones)
                if docvector.has_key(ngramnum):
                    docvector[ngramnum] += 1.0
                else:
                    docvector[ngramnum] = 1.0
    outfile = open(outdir+filename.replace('.rec', '.txt'), 'w')
    for key, value in docvector.items():
        outfile.write(str(key)+' '+str(value)+' '+str(math.sqrt(value))+'\n')        
    print 'Finished with file '+outdir+filename

#Read list of possible unigrams
unigramfile = open(unigramlistin, 'r')
for line in unigramfile:
    splitline = line.split(' ')
    for splits in splitline:
        allunigrams[normalizeFeature(splits)] = numofunigrams
    numofunigrams += 1

#Create training and devtest vectors
for i in range(len(setdir)):
    alldocs = []
    for language in languages:
        outdir = outbasedir.replace('?', language)+setdir[i]
        indir = inbasedir.replace('?', language)+setdir[i]
        os.system('mkdir -p '+outdir)
        for filename in os.listdir(indir):
            vectorize(indir, outdir, filename)
            alldocs.append(DocInfo(outdir+filename, language))
    outfile = open(outfilelistnames[i], 'w')
    for doc in alldocs:
        outfile.write(doc.lang+' '+doc.fname+'\n')

#Create evaluation/NIST vectors
alldocs = []
keyfile = open(nistkeyfile, 'r')
os.system('mkdir -p '+nistoutdir)
for line in keyfile:
    splitline = line.split()
    if not os.path.isfile(nistindir+splitline[0]+'.rec'):
        continue
    splitline = line.split(' ')
    vectorize(nistindir, nistoutdir, splitline[0]+'.rec')
    alldocs.append(DocInfo(nistoutdir+splitline[0]+'.rec', splitline[1]))
outfile = open(outfilelistnames[2], 'w')
for doc in alldocs:
    outfile.write(doc.lang+' '+nistoutdir+doc.fname+'\n')
    

print 'Finished, total number of unigrams: '+str(numofunigrams)+', trigrams: '+str(math.pow(numofunigrams, 3))