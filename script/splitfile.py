import os

sec = 30
splitLength_max = sec*0.9*10000000
targetdir = 'SpeechSec30'

languages = [ 'ARABIC_EGYPT', 'ENG_GENRL', 'ENG_SOUTH', 'FARSI', 'FRENCH_CAN', 'GERMAN', 'HINDI', 'JAPANESE', 'KOREAN', 'MANDARIN_M', 'MANDARIN_T', 'SPANISH', 'SPANISH_CAR', 'TAMIL', 'VIETNAMESE' ]
subdirs = [ 'devtest_raw', 'evltest_raw', 'train_raw' ]

for language in languages:
  for subdir in subdirs:
    testdir = './CallFriend/'+language+'/transcripts/'+subdir+'/'
    outdir = './CallFriend/'+language+'/Splitted30/'+subdir+'/'

    os.system('mkdir -p '+outdir)

    for filename in os.listdir(testdir):
      if filename.endswith('.rec'):
        file = open(testdir+filename, 'r')

        splitLength = 0
        fileNmbr = 0

        outfile = filename.replace('.rec', '_'+str(fileNmbr)+'.txt')
        fileOut = open(outdir+outfile, 'w')

        for line in file:
          splitline = line.split(' ')

          if (splitline[2] == 'pau' or splitline[2] == 'spk' or splitline[2] == 'int') and splitLength < splitLength_max:
            fileOut.write(line)

          elif (splitline[2] == 'pau' or splitline[2] == 'spk' or splitline[2] == 'int') and splitLength > splitLength_max:
            fileOut.write(line)
            fileOut.close()

            if splitLength > 35*10000000:
              print 'Longer than 35 sec!'

            fileNmbr += 1
            splitLength = 0
            outfile = filename.replace('.rec', '_'+str(fileNmbr)+'.txt')
            fileOut = open(outdir+outfile, 'w')

          else:
            fileOut.write(line)
            splitLength += int(splitline[1]) - int(splitline[0])

        os.system('rm '+outdir+outfile)   #Delete the last splitted file, since it most probalby does not contain 30 sec of speech
        print 'Finished splitting file '+filename+' in '+language+'/'+subdir+' ('+str(fileNmbr)+' files)'
        
print 'finished'        