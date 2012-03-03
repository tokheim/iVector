import os

phnrec = 'HU'
subdirs = [ '3', '10', '30' ]

configdir = './PHN_HU_SPDAT_LCRC_N1500'
sounddir = './NIST/2003/lid03e1/raw/'
targetdir = './NIST/2003/lid03e1/transcripts/'



for subdir in subdirs:
    n = 0
    tot = len(os.listdir(sounddir+subdir))

    os.system('mkdir -p '+targetdir+subdir)

    for filename in os.listdir(sounddir+subdir):
        if filename.endswith(".raw"):

            outfile = filename.replace('.raw', '.rec')
            os.system('phnrec -c '+configdir+' -i '+sounddir+subdir+'/'+filename+' -o '+targetdir+subdir+'/'+outfile)

            n+=1
        print str(n)+' of '+str(tot)+' files in '+subdir+' done\n'

print 'finished'
