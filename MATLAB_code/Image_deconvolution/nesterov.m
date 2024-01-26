clear all;
close all;

addpath([pwd,'/functions']);

%%% initialize the random number generator to make the results repeatable
rng('default');
%%% initialize the generator using a seed of 1
rng(1);

%% Setup experiment
xtrue=double(imread('cman.png'))/255.0; % Cameraman image to be used for the experiment
figure(1), imshow(xtrue);
[nRows, nColumns] = size(xtrue); % size of the image


%%%% function handle for uniform blur operator (acts on the image
%%%% coefficients)
h = [1 1 1];
lh = length(h);
h = h/sum(h);
h = [h zeros(1,length(xtrue)-length(h))];
h = cshift(h,-(lh-1)/2);
h = h'*h;


%%% operators A and A'
H_FFT = fft2(h);
HC_FFT = conj(H_FFT);

A = @(x) real(ifft2(H_FFT.*fft2(x))); % A operator
AT = @(x) real(ifft2(HC_FFT.*fft2((x)))); % A transpose operator
ATA = @(x) real(ifft2((HC_FFT.*H_FFT).*fft2((x)))); % AtA operator

% generate 'y'
y = A(xtrue);
BSNRdb = 40; % we will use this noise level
sigma = sqrt(var(y(:)) / 10^(BSNRdb/10));
sigma2 = sigma^2;
y = y + sigma*randn(nRows, nColumns); % we generate the observation 'y'

figure(2), imshow(y)

%%% Algorithm parameters
lambda = sigma2; %%% regularization parameter
alpha = 300; %0.044; %%% hyperparameter of the prior
eps = 0.15;

% Lipschitz Constants
Lf = 1/sigma2; %%% Lipschitz constant of the likelihood
Lg = alpha; %%% Lipschitz constant of the prior ||x||^2
%Lg = 1/sigma2; %%%  Lipschitz constant of the prior TV
%Lg = 2.8657/eps^2; %%%  Lipschitz constant of the prior GMM
Lfg = Lf + Lg; %%% Lipschitz constant of the model

% Gradients, proximal and \log\pi trace generator function
proxG = @(x) chambolle_prox_TV_stop(x, 'lambda',alpha*lambda,'maxiter',25);
ATy = AT(y);
gradF = @(x) (ATA(x) - ATy)/sigma2; %%% gradient of the likelihood
gradG = @(x) alpha*x; %%% gradient of the prior ||x||^2
%gradG = @(x) (x -proxG(x))/lambda; %%% gradient of the prior TV
%gradG = @(x) 1/eps^2 * gmm_fun(x); %%% gradient of the prior GMM

gradU = @(x) gradF(x) + gradG(x); %%% gradient of the model
logPi = @(x) -(norm(y-A(x),'fro')^2)/(2*sigma2) -0.5*alpha*norm(x,2)^2; %%% logpi ||X||^2
%logPi = @(x) -(norm(y-A(x),'fro')^2)/(2*sigma2) -alpha*TVnorm(x); %%% logpi TV
%logPi = @(x) - (norm(y-A(x), 'fro')^2)/(2*sigma2);
%logPi = @(x) - (norm(y-A(x), 'fro')^2)/(2*sigma2) - sum(sum(x/(2*eps^2)*(gmm_fun(x))));



gradient_descent = 1;

L = Lf + alpha;
mu = alpha;
kappa = L/mu;



Niter=1500;
x=y;
y1=y;
factor=(sqrt(kappa)-1)/(sqrt(kappa)+1);

energy(1)=logPi(x)
mse(1)= immse(x,xtrue)
for i=1:Niter
    h=1/L;
    x1=y1-h*gradU(y1);
    y1=x1+factor*(x1-x);
    x=x1;
    energy(i+1)=logPi(x);
    mse(i+1) = immse(xtrue,x);
end
    figure(4), plot([0:1:Niter],mse) 
    figure(5), plot([0:1:Niter],energy)
    figure(3), imshow(x)
    figure(6), semilogy([0:1:Niter], abs(energy) )

    mse(end)
    psnr(x, xtrue)

% if gradient_descent
%     %h=0.9/Lfg;
%     h = 2 / (L + mu);
%     N=10000;
%     z=y;
%     
%     energy(1)=logPi(z)
%     mse(1)= immse(z,x)
%     for i=1:N
%         z=z-h*gradU(z);
%         energy(i+1)=logPi(z);
%         mse(i+1) = immse(x,z);
%     end
%     
%     figure(3), imshow(z)
%     
%     
%     figure(4), plot([0:1:N],energy)
%     figure(5), plot([0:1:N],mse)
% 
% end