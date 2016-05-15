from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#%matplotlib inline
import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs
from PIL import Image
#import image class


#code from here
import glob
imageFolderPath = "jaffe"
imagePath = glob.glob(imageFolderPath + "/*.tiff")
no_images = len(imagePath)
print no_images
Ims = np.empty([no_images,256,256])
### Above is to import images from file

for i in range(no_images):
    im = np.asarray(Image.open(imagePath[i]))
    Ims[i] = im
X = np.reshape(Ims,[213,65536])
X = X.T


psi = X.mean(1)# Get mean face
Psi_reshaped = np.reshape(psi,[256,256])
#plt.imshow(Psi_reshaped,cmap = cm.Greys_r)
#plt.show()

Psi = np.repeat(np.reshape(psi,[65536,1]),no_images,axis=1)

Fi = X - Psi
A = Fi
C = np.dot(A.T,A)
print C.shape
w,v = np.linalg.eigh(C)
# eigenvalue w is in ascending order, so the larger value is in the bottom
V = np.fliplr(v)

M = no_images
V_M = V
U = np.dot(A,V_M)
U_sum = np.sum(U,axis = 0)
U_sum = np.repeat(np.reshape(U_sum,[1,M]),65536,axis=0)
print U_sum.shape
U_normalized = np.divide(U,U_sum)#normalize eigenvectors
print U_normalized.shape

for i in range(205,212):
    
    plt.imshow(np.reshape(U_normalized[:,i],[256,256]),cmap = cm.Greys_r)
    plt.show()
    
Fi_hat = np.reshape(psi,[65536,1])

## Below is the reconstruction procedure
K=20
Omega_k = np.empty([K,no_images])
for i in range(no_images):
    for j in range(K):
        Wj = np.dot(U_normalized[:,j].T,Fi[:,i])
        Omega_k[j][i] = Wj
        Fi_hat = Fi_hat + np.dot(Wj,np.reshape(U_normalized[:,j],[65536,1]))
Fi_hat_reshaped = np.reshape(Fi_hat,[256,256])
plt.imshow(Fi_hat_reshaped,cmap = cm.Greys_r)
plt.show()


# Below is the recognition test procedure
#
testFolderPath = "test"
testImage = glob.glob(testFolderPath + "/*.tiff")
no_test_images = len(testImage)
print no_test_images
Im_test = np.empty([no_test_images,256,256])

for i in range(no_test_images):
    im = np.asarray(Image.open(imagePath[i]))
    Im_test[i] = im
Xtest = np.reshape(Im_test,[no_test_images,65536])
Xtest = Xtest.T


for i in range(no_test_images):
    Omega = []
    Fi_test = Xtest[i]-psi
    for j in range(K):
        Wj = np.dot(U_normalized[:,j].T,Fi_test)
        print Wj
        Omega.append(Wj)
    Omega = np.asarray(Omega).T
    DIST = []
    for k in range(no_images):
        dist = np.linalg.norm(Omega-Omega_k[:,k])
        DIST.append(dist)
    idx = np.argmin(np.asarray(DIST))
    e = np.amin(np.asarray(DIST))

    print idx
    if e<10:
        print 'The face is recognized as face %i\n',idx
    else:
        print 'There is no face matched\n'

