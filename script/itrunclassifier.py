'''
Created on Mar 12, 2012

@author: Tokminator

Language/dialect mapping:
If a language has two dialects, then 

Prove train med -e 0.001 for bedre resultat

'''
import subprocess
import os

dataPath = './iVectors/iterate/it?0?1?2'
sets = ['Rtrain', 'Rdevtest']
iterations = [str(i) for i in range(8)]
branchdepth = [str(i) for i in range(1, 5)]

def createPath(path, set, it, bd):
    return path.replace('?0', it).replace('?1', set).replace('?2', bd)

def doClassification(cmd):
    f = subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)
    line = ''
    while True:
        last_line = line
        #print 'Runclassifier: '+line
        line = f.stdout.readline()
        if not line: break
    res = float(last_line)
    #print 'result '+c+'%'
    return res

outFile = open('./itres.txt', 'w')
for it in iterations:
    for bdtrain in branchdepth:
        best_result = 0.0
        best_bdtrain = '-1'
        best_bdtest = '-1'
        trainPath = createPath(dataPath, sets[0], it, bdtrain)
        for bdtest in branchdepth:
            testPath = createPath(dataPath, sets[1], it, bdtest)
            
            res = doClassification('python ./runclassifier.py -r 0 -t '+trainPath+' -e '+testPath)
            
            if res > best_result:
                best_bdtrain = bdtrain
                best_bdtest = bdtest
                best_result = res
            
            if bdtrain == bdtest:
                print 'it '+it+' bd '+bdtrain+': '+str(res)
                outFile.write('it '+it+' bd '+bdtrain+': '+str(res))
        print 'Best it '+it+' bdtrain '+best_bdtrain+' bdtest '+best_bdtest+': '+str(best_result)
        outFile.write('Best it '+it+' bdtrain '+best_bdtrain+' bdtest '+best_bdtest+': '+str(best_result))