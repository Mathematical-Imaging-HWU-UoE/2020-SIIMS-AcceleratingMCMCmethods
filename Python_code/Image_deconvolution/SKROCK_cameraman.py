"""
IMAGE DEBLURRING EXPERIMENT - CAMERAMAN TEST IMAGE            
We implement the SK-ROCK algorithm described in: "Accelerating
Proximal Markov Chain Monte Carlo by Using an Explicit Stabilized
Method", Marcelo Pereyra, Luis Vargas Mieles, and Konstantinos C.
Zygalakis, SIAM Journal on Imaging Sciences, Vol. 13, No. 2, 2020
Permalink: https://doi.org/10.1137/19M1283719   
@author: Luis Vargas Mieles
"""
#%%
# initialize the random number generator to make the results repeatable
# initialize the generator using a seed of 10
import random
random.seed(10)

import sys
sys.path.append("C:/Users/teresa-klatzer/code/2020-SIIMS-AcceleratingMCMCmethods/Python_code/Image_deconvolution/functions")
import numpy as np
from skimage import data
from skimage.transform import resize
from chambolle_prox_TV import chambolle_prox_TV
from TVnorm import TVnorm
from SKROCK import SKROCK
from plot_results_SKROCK import plot_results_SKROCK
from cshift import cshift
from tqdm import tqdm
import time

#%% Setup experiment
N=256 # image dimension
x = data.camera() # Cameraman image to be used for the experiment
x = resize(x, (N,N),preserve_range=True,anti_aliasing=False) # 256x256 dim

# function handle for uniform blur operator (5x5)
h = np.array([1, 1, 1])
lh = len(h)
h = h/np.sum(h)
h = np.concatenate((h,np.zeros(N-lh)))
h = cshift(h,-(lh-1)/2)
h = h.reshape(-1,1) @ h.reshape(1,-1)

# H and H' operators in the fourier domain
H_FFT = np.fft.fft2(h)
HC_FFT = np.conj(H_FFT)
del h

# A operator
A = lambda x: np.real(np.fft.ifft2(np.multiply(H_FFT,np.fft.fft2(x))))
# A transpose operator
AT = lambda x: np.real(np.fft.ifft2(np.multiply(HC_FFT,np.fft.fft2(x))))
# AtA operator
ATA = lambda x: np.real(np.fft.ifft2(np.multiply(np.multiply(HC_FFT,H_FFT),np.fft.fft2((x)))))

# generate the blurred and noisy observation 'y'
y = A(x)
BSNR = 42 # we will use this noise level
sigma = np.sqrt(np.var(np.asarray(y).ravel()) / 10**(BSNR/10))
sigma2 = sigma**2
y = y + sigma*np.random.randn(N,N)

# Algorithm parameters
lambda_prox = sigma2 # regularization parameter
alpha = 0.044 # hyperparameter of the prior

# Lipschitz Constants
Lf = 1/sigma2 # Lipschitz constant of the likelihood
Lg = 1/lambda_prox # Lipshcitz constant of the prior
Lfg = Lf + Lg # Lipschitz constant of the model

# Gradients, proximal and \log\pi trace generator function
proxG = lambda x: chambolle_prox_TV(x,alpha*lambda_prox,25)
ATy = AT(y)
gradF = lambda x: (ATA(x) - ATy)/sigma2 # gradient of the likelihood
gradG = lambda x: (x -proxG(x))/lambda_prox # gradient of the prior
gradU = lambda x: gradF(x) + gradG(x) # gradient of the model
logPi = lambda x: -(np.linalg.norm(y-A(x))**2)/(2*sigma2) -alpha*TVnorm(x)

# SK-ROCK PARAMETERS
# number of internal stages 's'
nStagesROCK = 10
# fraction of the maximum step-size allowed in SK-ROCK (0,1]
percDeltat = 0.5

nSamplesBurnIn = 1#int(6e2) # number of samples to produce in the burn-in stage
nSamples = 1#int(2e3) # number of samples to produce in the sampling stage
XkSKROCK = y # Initial condition
logPiTrace=np.zeros(nSamplesBurnIn+nSamples)
logPiTrace[0]=logPi(XkSKROCK)
# to save the mean of the samples from burn-in stage
meanSamples_fromBurnIn = XkSKROCK
# to save the evolution of the MSE from burn-in stage
mse_fromBurnIn=np.zeros(nSamplesBurnIn+nSamples)
mse_fromBurnIn[0]=np.square(np.subtract(meanSamples_fromBurnIn,x)).mean()
# to save the mean of the samples in the sampling stage
meanSamples = np.zeros([N,N])
#to save the evolution of the MSE in the sampling stage
mse=np.zeros(nSamples)
#-------------------------------------------------------------------------

print(' ')
print('BEGINNING OF THE SAMPLING')

#-------------------------------------------------------------------------

print('Burn-in stage...')
progressBar = tqdm(total=nSamplesBurnIn-1)
start_exec = time.time()
for i in range(1,nSamplesBurnIn):
    # produce a sample using SK-ROCK
    XkSKROCK=SKROCK(XkSKROCK,Lfg,nStagesROCK,percDeltat,gradU)
    # save \log \pi trace of the new sample
    logPiTrace[i]=logPi(XkSKROCK)
    # mean
    meanSamples_fromBurnIn = (i/(i+1))*meanSamples_fromBurnIn \
        + (1/(i+1))*(XkSKROCK)
    # mse
    mse_fromBurnIn[i] = np.square(np.subtract(meanSamples_fromBurnIn,x)).mean()
    # update iteration progress bar
    progressBar.update(1)

print('\nEnd of burn-in stage')
progressBar.close()
#-------------------------------------------------------------------------
print('Sampling stage...') 
progressBar = tqdm(total=nSamples)  
for i in range(nSamples):
    # produce a sample using SK-ROCK
    XkSKROCK=SKROCK(XkSKROCK,Lfg,nStagesROCK,percDeltat,gradU)
    # save \log \pi trace of the new sample
    logPiTrace[i+nSamplesBurnIn]=logPi(XkSKROCK)
    # mean from burn-in stage
    meanSamples_fromBurnIn = \
       ((i+nSamplesBurnIn)/(i+nSamplesBurnIn+1))*meanSamples_fromBurnIn \
       + (1/(i+nSamplesBurnIn+1))*XkSKROCK
    # mse from burn-in stabe
    mse_fromBurnIn[i+nSamplesBurnIn] = np.square(np.subtract(meanSamples_fromBurnIn,x)).mean()
    # mean from sampling stage
    meanSamples = (i/(i+1))*meanSamples + (1/(i+1))*XkSKROCK
    # mse from sampling stage
    mse[i] = np.square(np.subtract(meanSamples,x)).mean()
    # update iteration progress bar
    progressBar.update(1)

end_exec = time.time()
progressBar.close()
print('END OF THE SK-ROCK SAMPLING')
print('Execution time of the SK-ROCK sampling: '+str(end_exec-start_exec)+' sec')

# %% Plot of the results
plot_results_SKROCK(y,x,nStagesROCK,meanSamples,logPiTrace,mse)
