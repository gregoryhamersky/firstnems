rec = loaded_rec
dir(rec)
rec.signals
signal = rec['resp']
data = signal.rasterize()._data
data,shape
data[2,1:1000:1]
cell = data[3,0:1000:1]
cell.shape   # should be 1000

plt.figure()
plt.plot(cell)
plt.xlabel('Time')

signal.chans[2]