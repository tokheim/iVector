'''
Created on Feb 22, 2012

@author: Tokminator
'''
import sys
import os
import math
import subprocess


#Setup
optionsValues = ''

sphereSquareConstant = 1.0
resultPath = './res.txt'
tempDir = './ivecttemp/'
modelDir = './svmmodels/'
tempTrainFileName = 'a'
tempTestFileName = 'b'
tempModelFileName = 'c'
tempResultFileName = 'd'
gridOptions = '-out '+tempDir+'e'
gridPath = './gridliblinear.py'

trainSymbol = '-t'
testSymbol = '-e'
resultsSymbol = '-r'
optionsSymbol = '-o'
numLanguages = 12

trainVectors = [];
testVectors = [];

class iVector:
    def __init__(self, language):
        self.lang = language
        self.ivect = []


def readIvectList(iList, path):
    vectorFile = open(path, 'r')
    for line in vectorFile:
        splitline = line.split(' ')
        vect = iVector(splitline[0])
        for i in range(1, len(splitline)):
            #Should always be numbered 1 to dim anyways
            vect.ivect.append(float(splitline[i].split(':')[1]))
        iList.append(vect)
    vectorFile.close()

def writeIvectList(iList, path):
    outfile = open(path, 'w')
    for item in iList:
        line = item.lang
        for i in range(len(item.ivect)):
            line += ' '+str(i+1)+':'+str(item.ivect[i])
        outfile.write(line+'\n')
    outfile.close()
        
def shiftMean(iList, means):
    for item in iList:
        for i in range(len(item.ivect)):
            item.ivect[i] -= means[i]

#Calculate and shift iVector features to mean zero
def calcMean(iList):
    means = [0.0]*len(iList[0].ivect)
    for item in iList:
        for i in range(len(item.ivect)):
            means[i] += item.ivect[i]
    for i in range(len(means)):
        means[i] = means[i]/len(iList)
    return means

#Should be called after shifting mean
def scale(iList, stdevs):
    for item in iList:
        for i in range(len(item.ivect)):
            item.ivect[i] = item.ivect[i] / stdevs[i]


def calcSquareDist(vector):
    length = 0.0
    for item in vector:
        length += item*item
    return length

#Ensures that a^T*b <= 1 where a and b are vectors, without loosing information
def applyStereoProj(iList):
    for item in iList:
        normTerm = math.sqrt(calcSquareDist(item.ivect)+sphereSquareConstant)
        for i in range(len(item.ivect)):
            item.ivect[i] = item.ivect[i]/normTerm
        item.ivect.append(math.sqrt(sphereSquareConstant)/normTerm)

def giveUnitLength(iList):
    for item in iList:
        normTerm = math.sqrt(calcSquareDist(item.ivect))
        for i in range(len(item.ivect)):
            item.ivect[i] = item.ivect[i]/normTerm

#Calculate and scale by standard deviation
def calcStdev(iList):
    var = [0.0]*len(iList[0].ivect)
    for item in iList:
        for i in range(len(item.ivect)):
            var[i] += item.ivect[i]*item.ivect[i]
    for i in range(len(var)):
        var[i] = math.sqrt(var[i]/(len(iList)-1))
    return var
    
#Interfaces gridliblinear (From liblinears easy.py)
def gridSearch(trainPath):
    cmd = gridPath+' '+gridOptions+' '+trainPath
    f = subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)
    line = ''
    while True:
        last_line = line
        print 'GridSearch: '+line
        line = f.stdout.readline()
        if not line: break
    #c,g,rate = map(float,last_line.split())
    c = last_line.split()[0]
    print 'C value parsed is '+c
    return c

#Saves means and stdevs
def writeScale(means, stdevs, savePath):
    outFile = open(savePath, 'w')
    meanLine = str(means[0])
    varLine = str(stdevs[0])
    for i in range(1, len(means)):
        meanLine += ' '+str(means[i])
        varLine += ' '+str(stdevs[i])
    outFile.write(meanLine+'\n'+varLine+'\n')
    outFile.close()

#Reads and scales iVectors in iList with scaling data
def readConfAndScale(fullPath, iList):
    inFile = open(fullPath, 'r')
    meanLine = inFile.readline().split(' ')
    varLine = inFile.readline().split(' ')
    means = [0.0]*len(meanLine)
    stdevs = [0.0]*len(varLine)
    for i in range(len(means)):
        means[i] = float(meanLine[i])
        stdevs[i] = float(varLine[i])
    shiftMean(iList, means)
    scale(iList, stdevs)
    
def main():
    trainPaths = []
    global optionsValues
    global trainVectors
    global testVectors
    #Read input parameters
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == trainSymbol:
            trainPaths.append(sys.argv[i+1].lower())
            #readIvectList(trainVectors, sys.argv[i+1])
        elif sys.argv[i] == testSymbol:
            readIvectList(testVectors, sys.argv[i+1])
        elif sys.argv[i] == resultsSymbol:
            resultPath = sys.argv[i+1]
        elif sys.argv[i] == optionsSymbol:
            optionsValues = sys.argv[i+1]
    trainPaths.sort()
    outname = ''.join(trainPaths).replace('/', '').replace('.','')
    
    #Force recalculation of model
    #os.system('rm '+modelDir+outname+'.model')
    
    os.system('mkdir '+tempDir)
    if not (os.path.exists(modelDir+outname+'.model') and os.path.exists(modelDir+outname+'.conf')):
        print 'Has to train new model'
        #This set of training docs haven't been used before
        for paths in trainPaths:
            readIvectList(trainVectors, paths)
        means = calcMean(trainVectors)
        shiftMean(trainVectors, means)
        shiftMean(testVectors, means)
        stdevs = calcStdev(trainVectors)
        scale(trainVectors, stdevs)
        scale(testVectors, stdevs)
        print 'Sets scaled'
        
        applyStereoProj(trainVectors)
        applyStereoProj(testVectors)
        #giveUnitLength(trainVectors)
        #giveUnitLength(testVectors)
        print 'Vector lengths normalized'
        
        
        writeScale(means, stdevs, modelDir+outname+'.conf')
        writeIvectList(trainVectors, tempDir+tempTrainFileName)
        print 'Sets saved'
        c_value = gridSearch(tempDir+tempTrainFileName)
        print 'Gridsearch finished'
        os.system('train '+optionsValues+' -c '+c_value+' '+tempDir+tempTrainFileName+' '+modelDir+outname+'.model')
        print 'Training finished'
    else:
        print 'Found previous model'
        readConfAndScale(modelDir+outname+'.conf', testVectors)
        print 'TestVectors scaled'
        
        applyStereoProj(testVectors)
        #giveUnitLength(testVectors)
        print 'Testvector lengths normalized'
    
    
    #Write files
    writeIvectList(testVectors, tempDir+tempTestFileName)
    os.system('predict '+tempDir+tempTestFileName+' '+modelDir+outname+'.model '+tempDir+tempResultFileName)
    print 'Testing done'
    
    #Check results (Hard value)
    guess = [ [0 for i in range(numLanguages)] for i in range(numLanguages)]
    inFile = open(tempDir+tempResultFileName, 'r')
    i = 0
    for line in inFile:
        guess[int(testVectors[i].lang)-1][int(line)-1] += 1
        i+=1
    inFile.close()
    
    
    for i in range(len(guess)):
        print str(guess[i])
    
    os.system('rm -r '+tempDir)
main()