import scipy.io
import numpy
from sklearn import svm
from matplotlib import pylab



data = scipy.io.loadmat("data.mat")
HandWriting = data['X']
TrueVal = data['y'][:,0]
nsize = TrueVal.shape[0]/10
fig = pylab.figure()

#manipulate image
def SimplifyWriting(xx):
    for i in range(0,xx.shape[0]):
        tmp = xx[i] - numpy.std(xx[i])
        xx[i] = (tmp > 0).astype(int)

trainingx = HandWriting[0:350]
crossx = HandWriting[350:450]
testx = HandWriting[450:500]
trainingy = TrueVal[0:350]
crossy = TrueVal[350:450]
testy = TrueVal[450:500]
for i  in range(1, 10):
    nbegin = i*nsize
    sec1 = nbegin+nsize*0.7
    sec2 = nbegin+nsize*0.9
    nend = (i+1)*nsize
    print(nbegin, nend)
    trainingx = numpy.vstack((trainingx, HandWriting[nbegin:sec1]))
    crossx = numpy.vstack((crossx, HandWriting[sec1:sec2]))
    testx = numpy.vstack((testx, HandWriting[sec2:nend]))
    trainingy = numpy.append(trainingy, TrueVal[nbegin:sec1])
    crossy = numpy.append(crossy, TrueVal[sec1:sec2])
    testy = numpy.append(testy, TrueVal[sec2:nend])

ShiftedWriting = AddTranslation(trainingx[0])
for i in range(1,trainingx.shape[0]):
    ShiftedWriting = numpy.vstack((ShiftedWriting,AddTranslation(trainingx[i])))
ShiftedTarget = numpy.repeat(trainingy,25)

SmallWriting = ShrinkWriting(trainingx[0])
for i in range(1,trainingx.shape[0]):
    SmallWriting = numpy.vstack((SmallWriting,ShrinkWriting(trainingx[i])))

SmallTestx = ShrinkWriting(testx[0])
for i in range(1,testx.shape[0]):
    SmallTestx = numpy.vstack((SmallTestx,ShrinkWriting(testx[i])))

clf = svm.SVC()
clf.fit(trainingx, trainingy)
clf.fit(ShiftedWriting, ShiftedTarget)
clf.fit(SmallWriting, trainingy)
predicty = clf.predict(testx)
predicty = clf.predict(SmallTestx)
print(sum(predicty - testy != 0)/testy.shape[0])
for i in range(0,testy.shape[0]):
    if(predicty[i] != testy[i]):
        print(i,predicty[i],testy[i])

def ShowWriting(xxxx):
    nx = numpy.sqrt(xxxx.shape[0]).astype(int)
    ImgArray = xxxx.reshape(nx, nx, order = 'F')
    pylab.imshow(ImgArray,cmap = 'Greys')
    pylab.colorbar()
    pylab.show()

def AddTranslation(Img1):
    Shift = Img1
    nx2 = Img1.shape[0]
    nx = numpy.sqrt(nx2).astype(int)
    for x in range(-2,3):
        for y in range(-2,3):
            if(x == 0 and y == 0) :
                continue
            Img2D = Img1.reshape(nx, nx, order = 'F')
            if(x > 0):
                x1 = x
                x2 = nx
            else:
                x1 = 0
                x2 = nx + x
            if(y > 0):
                y1 = y
                y2 = nx
            else:
                y1 = 0
                y2 = nx + y
            tmp = numpy.zeros((nx, nx))
            tmp[x1-x:x2-x,y1-y:y2-y] = Img2D[x1:x2,y1:y2]
            Shift = numpy.vstack((Shift,tmp.reshape(nx2, order = 'F')))
    return Shift



def ShrinkWriting(Img1):
    nx = numpy.sqrt(Img1.shape[0]).astype(int)
    tmp = Img1.reshape(nx, nx)[::2, 1::2]
    return tmp.reshape(nx*nx/4)
