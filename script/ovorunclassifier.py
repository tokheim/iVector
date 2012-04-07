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

def writeIvectList(iList, path, positiveClass, negativeClass = 0):
    outfile = open(path, 'w')
    for item in iList:
        line = ''
        if item.lang == positiveClass:
            line = '1'
        elif not negativeClass or item.lang == negativeClass:
            line = '-1'
        else:
            continue
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
        scores = evaluate(results, i)
        for j in range(maxLabel):
            line += ' '+str(scores[j][1])
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

def decide(scores, soft = 0):
    bestLabel = 0
    for i in range(1, maxLabel):
        if scores[i][soft] > scores[bestLabel][soft]:
            bestLabel = i
        elif scores[i][soft] == scores[bestLabel][soft] and scores[i][1] > scores[bestLabel][1]:#decides ties in hardscore
            bestLabel = i
    return bestLabel
        
    
def evaluate(results, doc):
    softScore = [0.0]*len(maxLabel)
    hardScore = [0.0]*len(maxLabel)#Hardscore is float since it has to be normalized to the number of ovo's performed for each class
    for posLabel in range(1, maxLabel):
        for negLabel in range(posLabel):
            if posLabel%numLanguages == negLabel%numLanguages:#Dialect
                continue
            softScore[posLabel] += results[posLabel][negLabel][doc]
            softScore[negLabel] += 1.0-results[posLabel][negLabel][doc]
            if results[posLabel][negLabel][doc] > 0.5:
                hardScore[posLabel]+=1
            else:
                hardScore[negLabel]+=1
    for i in range(0, maxLabel):
        if i <= maxLabel-numLanguages or >= numLanguages:#Has dialects
            softScore[i] = softScore[i]/(maxLabel-2)
            hardScore[i] = hardScore[i]/(maxLabel-2)
        else:
            softScore[i] = softScore[i]/(maxLabel-1)
            hardScore[i] = hardScore[i]/(maxLabel-1)
    return zip(hardScore, softScore)
    
def recognize(results, doc, soft = 0):
    return decide(evaluate(results, doc), hard)
    

#Finds the number of correctly identified utterances, when different c-indexes are used
def findCorrect(results, soft = 0):
    #results[posLabel][negLabel][document]
    correct = 0.0
    for i in range(len(testVectors)):
        label = recognize(results, i, soft)
        if (int(testVectors[i].lang)-1)%numLanguages == label%numLanguages:
            correct+=1.0
    return correct

def printCorrect(results, soft = 0):
    confMatrix = [[0 for _ in range(numLanguages)] for _ in range(numLanguages)]
    for i in range(len(testVectors)):
        highest = recognize(results, i, soft)
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
    outPath = modelDir+''.join(trainPaths).replace('/', '').replace('.','')+'ovo/'
    

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
        
        writeIvectList(testVectors, tempTestPath, 'null')#The labels doesn't have to be correct since we only care about classifier scores
        
       
        #Res holds classifier results
        #Res[c][posLabel][negLabel][document]
        res = [[[[0.0 for testDoc in range(len(testVectors))] for negLabel in range(posLabel)] for posLabel in range(maxLabel)] for c in range(len(c_values))]
        

        for posLabel in range(maxLabel):
            for negLabel in range(posLabel):
                if posLabel%numLanguages == negLabel%numLanguages:#Only different dialects
                    continue
                writeIvectList(trainVectors, tempTrainPath, str(posLabel+1), str(negLabel+1))
                for c in range(len(c_values)):
                    os.popen('train -s 0 -c '+c_values[c]+' '+tempTrainPath+' '+tempModelPath)
                    print 'Model for language '+str(posLabel+1)+' and '+str(negLabel+1)+' with C '+c_values[c]+' created'
                    os.popen('predict -b 1 '+tempTestPath+' '+tempModelPath+' '+tempResultPath)
                    parseResults(tempResultPath, res[c][posLabel][negLabel])

        print  '--Starting hard search for parameters--'
        best_c = -1
        max_corr = 0
        for i in range(len(c_values)):
            correct = findCorrect(res[i], 0)
            if correct > max_corr:
                max_corr = correct
                best_c = i
                print 'New best c '+c_values[i]+' correct: '+str(correct/len(testVectors))
            else:
                print 'Not best c '+c_values[i]+' correct: '+str(correct/len(testVectors))
        
        printCorrect(res[best_c], 0)
        
        print '--Saving models--'
        for posLabel in range(maxLabel):
            for negLabel in range(posLabel):
                if posLabel%numLanguages == negLabel%numLanguages:#Only different dialects
                    continue
                writeIvectList(trainVectors, tempTrainPath, str(posLabel+1), str(negLabel+1))
                for c in range(len(c_values)):
                    os.popen('train -s 0 -c '+c_values[best_c]+' '+tempTrainPath+' '+outPath+'Model'str(posLabel)+'_'+str(negLabel))
        
        print '--Saving results--'
        saveResults(resultPath, res[best_c])    
        
        print '--Testing soft models--'
        best_c = -1
        max_corr = 0
        for i in range(len(c_values)):
            correct = findCorrect(res[i], 1)
            if correct > max_corr:
                max_corr = correct
                best_c = i
                print 'New best c '+c_values[i]+' correct: '+str(correct/len(testVectors))
            else:
                print 'Not best c '+c_values[i]+' correct: '+str(correct/len(testVectors))
        
        printCorrect(res[best_c], 1)
        
    else:
        print 'Found previous model'
        readConfAndScale(outPath+'conf', testVectors)
        print 'TestVectors scaled'
        
        applyStereoProj(testVectors)
        #giveUnitLength(testVectors)
        print 'Testvector lengths normalized'
        
        #Write all files
        writeIvectList(testVectors, tempTestPath, 'a')
        
        #Res holds classifier results
        #Res[posLabel][negLabel][document]
        res = [[[0.0 for testDoc in range(len(testVectors))] for negLabel in range(posLabel)] for posLabel in range(maxLabel)]
        
        
        print 'Testing'
        for posLabel in range(maxLabel):
            for negLabel in range(posLabel):
                if posLabel%numLanguages == negLabel%numLanguages:#Only different dialects
                    continue
                
                os.popen('predict -b 1 '+tempTestPath+' '+outPath+'Model'+str(posLabel)+'_'+str(negLabel)+' '+tempResultPath)
                parseResults(tempResultPath, res[posLabel][negLabel])
        
        printCorrect(res, 0)
        saveResults(resultPath, res)
    
    
    os.system('rm '+tempDir+'*')
main()
