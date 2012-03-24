'''
Created on 1. mars 2012

@author: Me

Each unsplitted file is about 30 mins
'''
import os
import math


devtargetmicros = 30 * 10000000;#Target microseconds of each file for devtest data
devtimeslack = 0.1
devtargetFiles = 100#Target number of devtest files per language

shortTraintargetLength = 30
traintargetFiles = 250#Target number of train files per language
traintimeslack = 0.4

numoffilesinlanguage = 120

languages = [ 'ARABIC_EGYPT', 'ENG_GENRL', 'ENG_SOUTH', 'FARSI', 'FRENCH_CAN', 'GERMAN', 'HINDI', 'JAPANESE', 'KOREAN', 'MANDARIN_M', 'MANDARIN_T', 'SPANISH', 'SPANISH_CAR', 'TAMIL', 'VIETNAMESE' ]
subdirs = [ 'devtest_raw', 'evltest_raw', 'train_raw' ]

class DocLens:
    def __init__(self, path, fname, length):
        self.fname = fname
        self.path = path
        self.length = length

def isSilence(symbol):
    return symbol == 'sil' or symbol == 'pau'

def isNoise(symbol):
    return symbol == 'spk' or symbol == 'int'
        
def insertSorted(docfile, doclist):
    for i in range(len(doclist)):
        if (doclist[i].length > docfile.length):
            doclist.insert(i, docfile)
            return
    doclist.append(docfile)

def writefile(fullpath, linelist, toindex):
    linelist.reverse()
    outfile = open(fullpath, 'w')
    for i in range(toindex):
        outfile.write(linelist.pop())
    outfile.close()
    linelist.reverse()


def splitfile(docfile, outdir, targetlength, targetslack):
    infile = open(docfile.path+docfile.fname)
    linelist = []
    lastpauseindex = 0
    lastpausetime = 0
    lastsplittime = 0
    num = 0
    outbase = outdir+docfile.fname.replace('.rec', '')+'_'
    for line in infile:
        splitline = line.split(' ')
        if len(splitline) > 3:
            if isSilence(splitline[2]) or isNoise(splitline[2]):
                if lastsplittime > targetlength and lastsplittime - targetlength < targetlength - lastpausetime:
                    #This is the optimal splitting time
                    writefile(outbase+str(num)+'.txt', linelist, len(linelist))
                    lastsplittime = 0
                    num+=1
                elif lastsplittime > targetlength and lastpausetime > (1-targetslack)*targetlength:
                    #Already found optimal splitting time
                    writefile(outbase+str(num)+'.txt', linelist, lastpauseindex)
                    lastsplittime -= lastpausetime
                    num+=1
                lastpauseindex = len(linelist)
                linelist.append(line)
                lastpausetime = lastsplittime
            else:
                if lastsplittime > (1+targetslack)*targetlength and lastpausetime > (1-targetslack)*targetlength:
                    #Already found optimal splitting time
                    writefile(outbase+str(num)+'.txt', linelist, lastpauseindex)
                    lastpauseindex = 0
                    lastsplittime -= lastpausetime
                    lastpausetime = 0
                    num+=1
                elif lastsplittime > (1+targetslack)*targetlength:
                    #No pause within slack, force split here
                    writefile(outbase+str(num)+'.txt', linelist, len(linelist))
                    lastpauseindex = 0
                    lastpausetime = 0
                    lastsplittime = 0
                    num+=1
                linelist.append(line)
                lastsplittime += int(splitline[1])-int(splitline[0])
    if (lastsplittime > (1-targetslack)*targetlength):
        #Last part of file is long enough to be included
        writefile(outbase+str(num)+'.txt', linelist, len(linelist))
        num+=1
    return num
    

for language in languages:
    fileListArray = []
    
    train_outdir = './CallFriend/' + language + '/vectsplit/train/'
    short_train_outdir = './CallFriend/'+language+ '/vectsplit/shorttrain/'
    dev_outdir = './CallFriend/' + language + '/vectsplit/devtest/'
    
    
    os.system('mkdir -p '+train_outdir)
    os.system('mkdir -p '+dev_outdir)
    os.system('mkdir -p '+short_train_outdir)
    os.system('rm '+train_outdir+'*.*')
    os.system('rm '+dev_outdir+'*.*')
    os.system('rm '+short_train_outdir+'*.*')
    
    
    for subdir in subdirs:
        trandir = './CallFriend/' + language + '/transcripts/' + subdir + '/'
        for filename in os.listdir(trandir):
            if not filename.endswith('.rec'):
                continue
            
            #Find the length of each file
            infile = open(trandir+filename, 'r')
            length = 0
            for line in infile:
                splitline = line.split(' ')
                if len(splitline) > 3 and not isSilence(splitline[2]) and not isNoise(splitline[2]):
                    length += int(splitline[1]) - int(splitline[0])
            infile.close()
            insertSorted(DocLens(trandir, filename, length), fileListArray)
    print 'Read length of '+language+' files'
    
    #Find and write devtest files
    numFiles = 0
    while numFiles < devtargetFiles:
        docfile = fileListArray[len(fileListArray)-1]
        if numFiles + docfile.length/devtargetmicros > devtargetFiles:#Find better suited files
            for dfile in fileListArray:
                if numFiles + dfile.length/devtargetmicros > devtargetFiles:
                    docfile = dfile
                    break
        fileListArray.remove(docfile)
        numFiles += splitfile(docfile, dev_outdir, devtargetmicros, devtimeslack)
        print 'Written '+str(numFiles)+' devtest files for '+language
        
    timeleft = 0
    for docfile in fileListArray:
        timeleft += docfile.length
    
    #Make short training files
    numFiles = 0.0
    shortList = list(fileListArray)
    while len(shortList) > 0:
        docfile = shortList.pop()
        numFiles += splitfile(docfile, short_train_outdir, shortTraintargetLength, traintimeslack)
        print str(len(shortList))+' short files for '+language+' remaining, currently split to '+str(numFiles)+' files'
    
    #Make train files
    numFiles = 0.0
    while len(fileListArray) > 0:
        docfile = fileListArray.pop()
        splits = max(round((traintargetFiles-numFiles)*docfile.length/(timeleft+1)), 1)
        numFiles += splitfile(docfile, train_outdir, docfile.length/splits, traintimeslack)
        timeleft -= docfile.length
        print str(len(fileListArray))+' files for '+language+' remaining, currently split to '+str(numFiles)+' files'
