import math
import os


names = ['Rtrain', 'Rdevtest', 'Revltest']
num = '9'
indir = '../src/iVectors/nomap50noLimitReset/'
testmodel = 'testmodel'
testresults = 'testresults'

svmtrain = '../libsvm-3.11/svm-train '
svmpredict = '../libsvm-3.11/svm-predict '

numLanguages = 13

labels = []

c_values = [str(math.pow(2, i)) for i in range(-5, 15, 2)]
g_values = [str(math.pow(2, i)) for i in range(3, -15, -2)]


def calccorrect(resdir, corlabels):
    infile = open(resdir, 'r')
    correct = 0.0
    tot = 0
    for line in infile:
        splitline = line.split(' ')
        if int(splitline[0].rstrip()) == int(corlabels[tot]):
            correct += 1.0
        tot += 1
    return correct/tot

for name in names:
    infile = open(indir+name+num, 'r')
    outfile = open('./'+name, 'w')
    setlabel = []
    for line in infile:
        splitline = line.split(' ')
        if int(splitline[0]) > numLanguages:
            outfile.write(str(int(splitline[0])-numLanguages))
            setlabel.append(int(splitline[0])-numLanguages)
        else:
            outfile.write(splitline[0])
            setlabel.append(int(splitline[0]))
        for i in range(1, len(splitline)):
            outfile.write(' '+splitline[i])
        #outfile.write('\n')
    labels.append(setlabel)
    outfile.close()
    infile.close()
print 'prep done'


bestC = 'tull'
bestG = 'tull'
bestRes = -100
for c_val in c_values:
    for g_val in g_values:
        os.popen(svmtrain+' -q -g '+g_val+' -c '+c_val+' '+names[0]+' '+testmodel)
        os.popen(svmpredict+names[1]+' '+testmodel+' '+testresults)
        tempres = calccorrect(testresults, labels[1])
        if tempres > bestRes:
            bestRes = tempres
            bestC = c_val
            bestG = g_val
            print 'new best '+c_val+' '+g_val+' res '+str(bestRes)
        else:
            print 'not best '+c_val+' '+g_val+' res '+str(tempres)+' best '+str(bestRes)
