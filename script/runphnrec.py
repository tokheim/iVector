import os

language = 'ENG_GENRL'
subdirs = [ 'evltest_raw', 'devtest_raw', 'train_raw' ]

configdir = './PHN_HU_SPDAT_LCRC_N1500'
sounddir = '/talebase/data/speech_raw/CallFriend/'+language+'/data/'
targetdir = './CallFriend/'+language+'/transcripts/'


for subdir in subdirs:
  n = 0
  tot = len(os.listdir(sounddir+subdir))
  
  os.system('mkdir -p '+targetdir+subdir)

  for filename in os.listdir(sounddir+subdir):
    if filename.endswith(".raw"):
      
      outfile = filename.replace('.raw', '.rec')
      os.system('phnrec -c '+configdir+' -i '+sounddir+subdir+'/'+filename+' -o '+targetdir+subdir +'/'+outfile)
      
      n+=1
      print str(n)+' of '+str(tot)+' files in '+subdir+' done\n'
print 'finished'