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


sphereSquareConstant = 1.0
resultPath = './res.txt'
tempDir = '../ivecttemp/'
modelDir = './svmmodels/'
tempTrainPath = tempDir+'a'
tempTestPath = tempDir+'b'
tempModelPath = tempDir+'c'
tempResultPath = tempDir+'d'
forceTrain = 0



trainSymbol = '-t'
testSymbol = '-e'
resultsSymbol = '-R'
forceSymbol = '-f'
numLanguages = 13#Actual number of languages, not including dialects but including 1 out of set language (as last language)
maxLabel = 16#Highest class label

c_values = [str(math.pow(2, i)) for i in range(-2, 6)]

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

def writeIvectList(iList, path, positiveClass, skipClass = ''):
    outfile = open(path, 'w')
    for item in iList:
        if item.lang == positiveClass:
            line = '1'
        elif item.lang == skipClass:
            continue
        else:
            line = '-1'
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
            
#Saves result vector for further processing
def saveResults(resultPath, results):
    outFile = open(resultPath, 'w')
    for i in range(len(testVectors)):
        line = testVectors[i].lang
        for j in range(maxLabel):
            line += ' '+str(results[j][i])
        outFile.write(line+'\n')
    outFile.close()
    
def parseResults(resultPath, results):
    inFile = open(resultPath, 'r')
    splitLine = inFile.readline().split()
    posIndex = 2
    i = 0
    if splitLine[1] == '1':
        posIndex = 1
            
    for line in inFile:
        splitLine = line.split()
        results[i] = float(splitLine[posIndex])
        i += 1

#Finds the number of correctly identified utterances, when different c-indexes are used
def findCorrect(results, cIndexes):
    correct = 0
    for i in range(len(testVectors)):
        bestVal = 0.0
        bestLabel = -1
        for j in range(maxLabel):
            if results[cIndexes[j]][j][i] > bestVal:
                bestVal = results[cIndexes[j]][j][i]
                bestLabel = j
        if bestLabel % numLanguages == (int(testVectors[i].lang)-1) % numLanguages:
            correct += 1.0
    return correct

def printCorrect(results):
    confMatrix = [[0 for _ in range(numLanguages)] for _ in range(numLanguages)]
    for i in range(len(testVectors)):
        highest = 0
        for j in range(1, maxLabel):
            if results[j][i] > results[highest][i]:
                highest = j
        confMatrix[(int(testVectors[i].lang)-1)%numLanguages][highest%numLanguages] += 1
    
    correct = 0.0
    for i in range(numLanguages):
        print str(confMatrix[i])
        correct += confMatrix[i][i]
    print correct/len(testVectors)

def main():
    
    trainPaths = []
    
    global trainVectors
    global testVectors
    global regressionValue
    global resultPath
    global forceTrain
    #Read input parameters
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == trainSymbol:
            trainPaths.append(sys.argv[i+1].lower())
            #readIvectList(trainVectors, sys.argv[i+1])
        elif sys.argv[i] == testSymbol:
            readIvectList(testVectors, sys.argv[i+1])
        elif sys.argv[i] == resultsSymbol:
            resultPath = sys.argv[i+1]
        elif sys.argv[i] == forceSymbol:
            forceTrain = int(sys.argv[i+1])
    trainPaths.sort()
    outPath = modelDir+''.join(trainPaths).replace('/', '').replace('.','')+'/'
    

    os.system('mkdir -p '+tempDir)
    if not os.path.exists(outPath) or forceTrain:
        print 'Has to train new model'
        os.system('mkdir -p '+outPath)
        
        
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
        
        writeScale(means, stdevs, outPath+'conf')
        
        writeIvectList(testVectors, tempTestPath, 'null')#Scores are calculated in this script anyways...
        
        #Holds recognition results, dim1 = c-value, dim2 label, dim3 = testdoc
        res = [[[0.0 for _ in range(len(testVectors))] for _ in range(maxLabel)] for _ in range(len(c_values))]
        
        #Write all train models
        for i in range(1, maxLabel+1):
           
            if i == numLanguages:#Out of set language
                continue
            elif i <= maxLabel - numLanguages:#The language has dialects
                writeIvectList(trainVectors, tempTrainPath, str(i), str(i+numLanguages))
            elif i > maxLabel:#The language has dialects
                writeIvectList(trainVectors, tempTrainPath, str(i), str(i-numLanguages))
            else:
                writeIvectList(trainVectors, tempTrainPath, str(i))
                
            for j in range(len(c_values)):
                os.popen('train -s 0 -c '+c_values[j]+' '+tempTrainPath+' '+tempModelPath)
                print 'Model for language '+str(i)+' and C '+c_values[j]+' created'
                os.popen('predict -b 1 '+tempTestPath+' '+tempModelPath+' '+tempResultPath)
                parseResults(tempResultPath, res[j][i-1])
                
        print '--All models saved--'
        
        max_corr = 0
        best_c = 0
        for i in range(len(c_values)):
            correct = findCorrect(res, [i]*maxLabel)
            if correct > max_corr:
                max_corr = correct
                best_c = i
                print 'New best c '+c_values[i]+' correct: '+str(float(correct)/len(testVectors))
            else:
                print 'Not best c '+c_values[i]+' correct: '+str(float(correct)/len(testVectors))
        res1 = res[best_c]
        printCorrect(res1)
        
        for i in range(1, maxLabel+1):
            if i == numLanguages:#Out of set language
                continue
            elif i <= maxLabel - numLanguages:#There is a dialect
                writeIvectList(trainVectors, tempTrainPath, str(i), str(i+numLanguages))
            elif i > maxLabel:#There is a dialect
                writeIvectList(trainVectors, tempTrainPath, str(i), str(i-numLanguages))
            else:
                writeIvectList(trainVectors, tempTrainPath, str(i))
                
            os.popen('train -s 0 -c '+c_values[best_c]+' '+tempTrainPath+' '+outPath+'Model'+str(i))
        print 'Models saved'
        
        
        
        c_array = [0]*maxLabel
        print 'Number two'
        for i in range(maxLabel):
            max_corr = -999999999990
            best_c = 0
            for j in range(len(c_values)):
                correct = 0
                for k in range(len(testVectors)):
                    if (int(testVectors[k].lang)-1) % numLanguages == i % numLanguages and res[j][i][k] >= 0.5:
                        correct += 1
                    elif (int(testVectors[k].lang)-1) % numLanguages != i % numLanguages and res[j][i][k] >= 0.5:
                        correct -= 1
                if correct > max_corr:
                    best_c = j
                    max_corr = correct
                    c_array[i] = j
                    print 'Lang '+str(i)+' new best C '+c_values[j]+' correct: '+str(float(correct)/len(testVectors))
        
        print 'C used: '+str(c_array)
        res2 = [[res[c_array[j]][j][i] for i in range(len(testVectors))] for j in range(maxLabel)]
        printCorrect(res2)
        
        print 'Number three'
        for i in range(maxLabel):
            max_corr = -9999999999990.0
            best_c = 0
            for j in range(len(c_values)):
                correct = 0
                for k in range(len(testVectors)):
                    if (int(testVectors[k].lang)-1) % numLanguages == i % numLanguages:
                        correct += res[j][i][k]
                    elif (int(testVectors[k].lang)-1) % numLanguages != i % numLanguages:
                        correct -= res[j][i][k]
                if correct > max_corr:
                    best_c = j
                    max_corr = correct
                    c_array[i] = j
                    print 'Lang '+str(i)+' new best C '+c_values[j]+' correct: '+str(correct)
        
        res3 = [[res[c_array[j]][j][i] for i in range(len(testVectors))] for j in range(maxLabel)]
        print 'C used: '+str(c_array)
        printCorrect(res3)
        
        
        
        saveResults(resultPath, res1)    
        
        
        
    else:
        print 'Found previous model'
        readConfAndScale(outPath+'conf', testVectors)
        print 'TestVectors scaled'
        
        applyStereoProj(testVectors)
        #giveUnitLength(testVectors)
        print 'Testvector lengths normalized'
        
        res = [[0.0 for _ in range(len(testVectors))] for _ in range(maxLabel)]
        #Write files
        writeIvectList(testVectors, tempTestPath, 'a')
        for i in range(1, maxLabel+1):
            if i == numLanguages:#Out of set language
                continue
            os.popen('predict -b 1 '+tempTestPath+' '+outPath+'Model'+str(i)+' '+tempResultPath)
            parseResults(tempResultPath, res[i-1])
        printCorrect(res)
        saveResults(resultPath, res)
    
    
    os.system('rm '+tempDir+'*')
main()
