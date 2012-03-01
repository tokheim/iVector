import os


fromdirs = ['/talebase/data/speech_raw/NIST_LR/2007/data/lid03e1/test/10/', '/talebase/data/speech_raw/NIST_LR/2007/data/lid03e1/test/30/']
todirs = ['./NIST/2003/lid03e1/raw/10/', './NIST/2003/lid03e1/raw/30/']
soxdir = '~/sox-14.3.2/sox'

n = 0
for i in range(len(fromdirs)):
  os.system('mkdir -p '+todirs[i])
  
  for filename in os.listdir(fromdirs[i]):
    if filename.endswith('.sph'):
      outfile = filename.replace('.sph', '.raw')
      os.system(soxdir+' '+fromdirs[i]+filename+' -b 16 '+todirs[i]+outfile)
    else:
      os.system('cp '+fromdirs[i]+filename+' '+todirs[i]+filename)
    n+=1
    print str(n)+' done'
    
