import os

languages = [ 'ARABIC_EGYPT', 'ENG_GENRL', 'ENG_SOUTH', 'FARSI', 'FRENCH_CAN', 'GERMAN', 'HINDI', 'JAPANESE', 'KOREAN', 'MANDARIN_M', 'MANDARIN_T', 'SPANISH', 'SPANISH_CAR', 'TAMIL', 'VIETNAMESE' ]
inbasedir = './CallFriend/?/vectsplit/'
nistindir = './NIST/2003/lid03e1/transcripts/30/'
setdir = ['train/', 'devtest/']
microsinsecond = 10000000.0


def isSilence(symbol):
    return symbol == 'sil' or symbol == 'pau'

def isNoise(symbol):
    return symbol == 'spk' or symbol == 'int'

def readLength(filepath):
    #Find the length of each file
    infile = open(filepath, 'r')
    length = 0
    for line in infile:
        splitline = line.split(' ')
        if len(splitline) > 3 and not isSilence(splitline[2]) and not isNoise(splitline[2]):
            length += int(splitline[1]) - int(splitline[0])
    infile.close()
    return length/microsinsecond

for set in setdir:
    numfiles = 0.0
    length = 0
    for language in languages:
        indir = inbasedir.replace('?', language)+set
        for filename in os.listdir(indir):
            length += readLength(indir+filename)
            numfiles += 1
    print set+' files: '+str(numfiles)+' Length '+str(length)+' Avg length '+str(length/numfiles)

numfiles = 0.0
length = 0
for filename in os.listdir(nistindir):
    numfiles += 1
    length += readLength(nistindir+filename)
print 'NIST files: '+str(numfiles)+' Length '+str(length)+' Avg length '+str(length/numfiles)