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
import random


sphereSquareConstant = 1.0
resultPath = './res.txt'
tempDir = '../ivecttemp/'
modelDir = './svmmodels/'
tempTrainPath = tempDir+'a'
tempTestPath = tempDir+'b'
tempModelPath = tempDir+'c'
tempResultPath = tempDir+'d'
forceTrain = 1



trainSymbol = '-t'
testSymbol = '-e'
resultsSymbol = '-R'
forceSymbol = '-f'
numLanguages = 13#Actual number of languages, not including dialects but including 1 out of set language (as last language)
maxLabel = 16#Highest class label
numBits = 30

c_values = [str(math.pow(2, i)) for i in range(-1, 5)]

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

def writeIvectList(iList, path, posList = [0]*maxLabel):
    outfile = open(path, 'w')
    for item in iList:
        if posList[int(item.lang)-1]:
            line = '1'
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
        
#Odd parity huffman matrix, Matrix[bit][language]
def getHuffmanMatrix():
    matrix = [[0 for _ in range(maxLabel)] for _ in range(7)]
    for lang in range(maxLabel):
        for bit in range(4):
            matrix[bit][lang] = int((lang%math.pow(2, 4-bit))/math.pow(2, 3-bit))
        matrix[4][lang] = (1+matrix[0][lang]+matrix[1][lang]+matrix[2][lang])%2
        matrix[5][lang] = (1+matrix[0][lang]+matrix[2][lang]+matrix[3][lang])%2
        matrix[6][lang] = (1+matrix[1][lang]+matrix[2][lang]+matrix[3][lang])%2
    return matrix

#if distance is less than both breakdist (0,0) is returned also checks row complements distance
def getRowColDist(matrix, colBreakDist = 0, rowBreakDist = 0):
    minRowDist = len(matrix)
    minColDist = len(matrix[0])
    for row1 in range(1, len(matrix)):
        for row2 in range(row1):
            diffs = 0
            for col in range(len(matrix[0])):
                if matrix[row1][col] != matrix[row2][col]:
                    diffs += 1
            if diffs < minRowDist:
                minRowDist = diffs
            if len(matrix[row1])-diffs < minRowDist:
                minRowDist = len(matrix[row1])-diffs
                if rowBreakDist < colBreakDist and minRowDist < rowBreakDist:
                    return (0, 0)
                
    for col1 in range(1, len(matrix[0])):
        for col2 in range(col1):
            diffs = 0
            for row in range(len(matrix)):
                if matrix[row][col1] != matrix[row][col2]:
                    diffs += 1
            if diffs < minColDist:
                minColDist = diffs
                if minColDist <= colBreakDist and minRowDist <= rowBreakDist:
                    return (0,0)
    return (minRowDist, minColDist)
    
def isValid(matrix):
    for i in range(len(matrix)):
        hasZero = 0
        hasOne = 0
        for j in range(len(matrix[i])):
            if matrix[i][j] and j != numLanguages-1:
                hasOne = 1
                if hasZero:
                    break
            elif j != numLanguages-1:
                hasZero = 1
                if hasOne:
                    break
        if hasZero == 0 or hasOne == 0:
            return 0
    return 1

def getRandomMatrix(bits, threshold):
    return [[int(random.random() < 0.2) for _ in range(maxLabel)] for _ in range(bits)]

def getSparseMatrix(bits, onesPerRow):
    matrix = [[0 for _ in range(maxLabel)] for _ in range(bits)]
    for row in range(bits):
        ones = float(onesPerRow)
        for col in range(maxLabel-1, -1, -1):
            if random.random() < ones/(col+1):
                matrix[row][col] = 1
                ones -= 1
    return matrix
            
            

def createMatrixMatrix(bits):
    attempts = 200000 / bits
    random.seed(23)
    maxRowDist = 0
    maxColDist = 0
    bestMat = [[0]]
    for i in range(attempts):
        #matrix = [[int(random.random() > 0.2) for _ in range(maxLabel)] for _ in range(bits)]
        matrix = getSparseMatrix(bits, 5)
        if not isValid(matrix):
            continue
        
        dist = getRowColDist(matrix, maxColDist, maxRowDist)
        
        
        if (maxRowDist < maxColDist and dist[0] > maxRowDist) or (maxColDist < maxRowDist and dist[1] > maxColDist) or (dist[0] >= maxRowDist and dist[1] > maxColDist):
        #if min(dist[0], dist[1]) > min(maxRowDist, maxColDist) or (dist[0] >= maxRowDist and dist[1] >= maxColDist):
            bestMat = matrix
            maxRowDist = dist[0]
            maxColDist = dist[1]
            print 'New best matrix found with row dist '+str(dist[0])+' and col dist '+str(dist[1])
    return bestMat                                       

def findErrors(results, codeLang):
    errors = 0
    for i in range(len(testVectors)):
        lang = int(testVectors[i].lang)-1
        for bit in range(len(codeLang)):
            if codeLang[bit][lang] and results[bit][i] < 0.5:
                errors += 1
            elif not codeLang[bit][lang] and results[bit][i] > 0.5:
                errors += 1
    return errors

def findSoftErrors(results, codeLang):
    errors = 0.0
    for i in range(len(testVectors)):
        lang = int(testVectors[i].lang)-1
        for bit in range(len(codeLang)):
            if codeLang[bit][lang]:
                errors += 1-results[bit][i]
            else:
                errors += results[bit][i]
    return errors

#Returns a matrix with element [i][0] being the number of negative classes of classifier i, and [i][1] positive classes
def getSetNorms(codeLang):
    setNorms = [[0 for _ in range(2)] for _ in range(len(codeLang))]
    for bit in range(len(codeLang)):
        for lang in range(maxLabel):
            if lang == numLanguages-1:#Out of set
                continue
            if codeLang[bit][lang]:
                setNorms[bit][1] += 1
            else:
                setNorms[bit][0] += 1
    return setNorms

def printCorrect(results, codeLang):
    hardConfMatrix = [[0 for _ in range(numLanguages)] for _ in range(numLanguages)]
    softConfMatrix = [[0 for _ in range(numLanguages)] for _ in range(numLanguages)]
    softProdConfMatrix = [[0 for _ in range(numLanguages)] for _ in range(numLanguages)]
    
    setNorms = getSetNorms(codeLang)
    
    for i in range(len(testVectors)):
        #lang = int(testVectors[i].lang)-1
        bestHardDist = len(codeLang)
        bestHardIndex = -1
        bestSoftDist = len(codeLang)
        bestSoftIndex = -1
        bestSoftProd = 0
        bestSoftProdIndex = -1
        for lang in range(maxLabel):
            hardDist = 0.0
            softDist = 0.0
            softProd = 1.0
            for bit in range(len(codeLang)):
                softDist += math.fabs(codeLang[bit][lang]-results[bit][i])
                if results[bit][i] > 0.5:
                    hardDist += math.fabs(codeLang[bit][lang]-1)
                else:
                    hardDist += codeLang[bit][lang]
                    
                if codeLang[bit][lang]:
                    softProd *= results[bit][i]/setNorms[bit][1]
                else:
                    softProd *= (1-results[bit][i])/setNorms[bit][0]
                
            if hardDist < bestHardDist:
                bestHardDist = hardDist
                bestHardIndex = lang
            if softDist < bestSoftDist:
                bestSoftDist = softDist
                bestSoftIndex = lang
            if softProd > bestSoftProd:
                bestSoftProd = softProd
                bestSoftProdIndex = lang
        hardConfMatrix[(int(testVectors[i].lang)-1)%numLanguages][bestHardIndex%numLanguages] += 1
        softConfMatrix[(int(testVectors[i].lang)-1)%numLanguages][bestSoftIndex%numLanguages] += 1
        softProdConfMatrix[(int(testVectors[i].lang)-1)%numLanguages][bestSoftProdIndex%numLanguages] += 1
        
    
    hardCorrect = 0.0
    softCorrect = 0.0
    softProdCorrect = 0.0
    hardStr = ''
    softStr = ''
    softProdStr = ''
    for i in range(numLanguages):
        hardStr += str(hardConfMatrix[i])+'\n'
        softStr += str(softConfMatrix[i])+'\n'
        softProdStr += str(softProdConfMatrix[i])+'\n'
        hardCorrect += hardConfMatrix[i][i]
        softCorrect += softConfMatrix[i][i]
        softProdCorrect += softProdConfMatrix[i][i]
        
    print hardStr+'Hard res '+str(hardCorrect/len(testVectors))
    print softStr+'Soft res '+str(softCorrect/len(testVectors))
    print softProdStr+'Soft product res '+str(softProdCorrect/len(testVectors))



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
    outPath = modelDir+''.join(trainPaths).replace('/', '').replace('.','')+'huff/'
    
    #Two representations of huffman codes for easy representation of rows and columns
    #codeLang = getHuffmanMatrix()
    #codeLang = getTestMatrix()
    codeLang = createMatrixMatrix(numBits)
    
    print 'CodeLang, min dist '+str(getRowColDist(codeLang))+':'
    for i in range(len(codeLang)):
        print str(codeLang[i])

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
        
        writeIvectList(testVectors, tempTestPath)#Scores are calculated in this script anyways so labels aren't important...
        
        #Holds recognition results, dim1 = c-value, dim2 bit, dim3 testdoc
        res = [[[0.0 for _ in range(len(testVectors))] for _ in range(len(codeLang))] for _ in range(len(c_values))]
        
        for bit in range(len(codeLang)):
            writeIvectList(trainVectors, tempTrainPath, codeLang[bit])
            for i in range(len(c_values)):
                os.popen('train -s 0 -c '+c_values[i]+' '+tempTrainPath+' '+tempModelPath)
                print 'Model '+str(bit*len(c_values)+i+1)+' of '+str(len(codeLang)*len(c_values))+' created'
                os.popen('predict -b 1 '+tempTestPath+' '+tempModelPath+' '+tempResultPath)
                parseResults(tempResultPath, res[i][bit])
        
        minHardErrors = len(testVectors)*len(codeLang)
        minSoftErrors = len(testVectors)*len(codeLang)
        bestHardC = -1
        bestSoftC = -1
        
        print 'Parameter search'
        for i in range(len(c_values)):
            hardErrors = findErrors(res[i], codeLang)
            softErrors = findSoftErrors(res[i], codeLang)
            if hardErrors < minHardErrors:
                minHardErrors = hardErrors
                bestHardC = i
                print 'New best hard c '+c_values[i]+' avg bit errors: '+str(float(hardErrors)/len(testVectors))
            if softErrors < minSoftErrors:
                minSoftErrors = softErrors
                bestSoftC = i
                print 'New best soft c '+c_values[i]+' avg soft error: '+str(softErrors/len(testVectors))
        
        print '---Soft optimize---'
        printCorrect(res[bestSoftC], codeLang)
        print '---Hard optimize---'
        printCorrect(res[bestHardC], codeLang)
                
                
        print '--All models saved--'
        
                
        #os.popen('train -s 0 -c '+c_values[best_c]+' '+tempTrainPath+' '+outPath+'Model'+str(bit))
        print 'Models saved'
        
        
        
        #saveResults(resultPath, res1)    
        
        
        
    else:
        print 'Found previous model'
        readConfAndScale(outPath+'conf', testVectors)
        print 'TestVectors scaled'
        
        applyStereoProj(testVectors)
        #giveUnitLength(testVectors)
        print 'Testvector lengths normalized'
        
        '''
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
        '''
    
    
    os.system('rm '+tempDir+'*')
main()
