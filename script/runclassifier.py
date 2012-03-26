'''
Created on Feb 22, 2012

@author: Tokminator

Language/dialect mapping:
If a language has two dialects, then 

Prove train med -e 0.001 for bedre resultat

'''
import sys
import os
import math
import subprocess


#Setup
optionsValues = ''
regressionValue = 1#Use regression by default
c_value = '-1'

sphereSquareConstant = 1.0
resultPath = './res.txt'#NOT USED
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
resultsSymbol = '-R'#NOT USED
optionsSymbol = '-o'
regressionSymbol = '-r'
regularizationSymbol = '-c'
numLanguages = 13#Actual number of languages, not including dialects
ignoreLastLanguage = 0#Ignore out of set language

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
    

def readHardResults(resultPath, testList, numLanguages, printResults):
    #Check results (Hard value)
    guess = [ [0 for i in range(numLanguages)] for i in range(numLanguages)]
    inFile = open(resultPath, 'r')
    i = 0
    for line in inFile:
        splitline = line.split(' ')
        guess[(int(testList[i].lang)-1)%numLanguages][(int(splitline[0])-1)%numLanguages] += 1
        i+=1
    inFile.close()
    
    tot = 0
    correct = 0.0
    for i in range(len(guess)):
        if not (ignoreLastLanguage and i == numLanguages-1):
            if printResults:
                print str(guess[i])
            tot += sum(guess[i])
            correct += guess[i][i]
    if printResults:
        print str(correct/tot)
    return correct/tot

def findCValue(trainPath, tempModelPath, testPath, tempResultPath, numLanguages, testList):
    c_best = '-1'
    cor_best = 0
    c_values = [str(math.pow(2, i)) for i in range(-4, 2)]
    for c_value in c_values:
        os.system('train -s 0 '+optionsValues+' -c '+c_value+' '+trainPath+' '+tempModelPath)
        os.system('predict '+testPath+' '+tempModelPath+'.model '+tempResultPath)
        
        res = readHardResults(tempResultPath, testList, numLanguages, 0)
        if res > cor_best:
            print 'c: '+c_value+' result: '+str(res)+' Currently best'
            c_best = c_value
            cor_best = res
        else:
            print 'c: '+c_value+' result: '+str(res)+' Not best'
    print 'Regularizaiton search finished, best c '+c_best
    return c_best
        
#Saves result vector for further processing
def readSoftResults(tempResultPath, testList, numLanguages, resultPath):
    #1 line: labels 1 2 3 ...
    #next lines: <guessed class> <prob1> <prob2> <prob3> <prob4> <prob5>...
    
    i = 0
    inFile = open(tempResultPath, 'r')

    outFile = open(resultPath, 'w')
    for line in inFile.readlines()[1:]:#Ignore first line
        splitLine = line.split()
        outLine = str(testList[i].lang)
        for j in range(1, len(splitLine)):
            outLine += ' '+ splitLine[j]
        outFile.write(outLine)
        i+=1
    
    outFile.close()
    inFile.close()
    
    cdettot = 0.0
    for i in range(numLanguages):
        if targets[i] != 0:
            cdet = (pmissgiventarget[i]/targets[i]+pfalsealarmgivennontarget[i]/(len(testList)-targets[i]))/2
        else:
            cdet = 0.5*pfalsealarmgivennontarget[i]/(len(testList)-targets[i])
        print str(i)+' C_det: '+str(cdet)
        cdettot += cdet
    print 'Avg C_det: '+str(cdettot/numLanguages)
    return cdettot/numLanguages
    
def main():
    trainPaths = []
    global optionsValues
    global trainVectors
    global testVectors
    global regressionValue
    global c_value
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
        elif sys.argv[i] == regressionSymbol:
            regressionValue = int(sys.argv[i+1])
        elif sys.argv[i] == regularizationSymbol:
            c_value = sys.argv[i+1]
    trainPaths.sort()
    outname = ''.join(trainPaths).replace('/', '').replace('.','')+c_value
    if regressionValue:
        outname += 'r'
        
    
    #Force recalculation of model
    os.system('rm '+modelDir+outname+'.model')
    
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
        if float(c_value) > 0:
            print 'Starting Gridsearch'
            c_value = findCValue(tempDir+tempTrainFileName, tempDir+tempModelFileName, tempDir+tempTestFileName, tempDir+tempResultFileName, numLanguages, testVectors)
            print 'Gridsearch finished'
        if not regressionValue:
            os.system('train '+optionsValues+' -c '+c_value+' '+tempDir+tempTrainFileName+' '+modelDir+outname+'.model')
        else:
            os.system('train -s 0 '+optionsValues+' -c '+c_value+' '+tempDir+tempTrainFileName+' '+modelDir+outname+'.model')
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
    if not regressionValue:
        os.system('predict '+tempDir+tempTestFileName+' '+modelDir+outname+'.model '+tempDir+tempResultResultFileName)
        print 'Testing done'
    
        readHardResults(tempDir+tempResultFileName, testVectors, numLanguages, 1)
    else:
        os.system('predict -b 1 '+tempDir+tempTestFileName+' '+modelDir+outname+'.model '+tempDir+tempResultFileName)
        print 'Testing done'
        
        readSoftResults(tempDir+tempResultFileName, testVectors, numLanguages)
    
    os.system('rm -r '+tempDir)
main()