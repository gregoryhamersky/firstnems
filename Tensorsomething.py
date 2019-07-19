for jj in range(Joe.shape[1]):
    rolled = np.roll(Joe,-1)
    np.append(Joe,rolled,axis=1)







testjoe = np.arange(25)
testjoe = np.expand_dims(testjoe,axis=0)


for jj in range(testjoe.shape[1]):
    rolledtest = np.roll(testjoe,(-jj))
    if jj == 0:
        joeroll = rolledtest
    else:
        joeroll = np.concatenate((joeroll,rolledtest),axis=0)


#This works for a single TORC
for jj in range(Joe.shape[1]):
    rolledJoe = np.roll(Joe,(-jj))
    if jj == 0:
        JoeDelay = rolledJoe
    else:
        JoeDelay = np.concatenate((JoeDelay,rolledJoe),axis=0)





########################
#Create S* 375x25x30
DelayTorcs = dict()
for tt,(key,value) in enumerate(TorcValues.items()):
    for jj in range(value.shape[1]):
        rolledtor = np.roll(value,(-jj))
        if jj == 0:
            torcdel = rolledtor
        else:
            torcdel = np.concatenate((torcdel,rolledtor),axis=0)
    DelayTorcs[key] = torcdel
    if tt == 0:
        unwrap = torcdel
    else:
        unwrap = np.concatenate((unwrap,torcdel),axis=1)
stackedtorc = np.stack(list(DelayTorcs.values()), axis=2)

stimall = np.tile(unwrap,10)
########################

#Create r* 750x1 (x10,keys)##
for cc, (key,value) in enumerate(RespCat.items()):
    if cc == 0:
        respall = value
    else:
        respall = np.concatenate((respall,value),axis=0)
#############################

