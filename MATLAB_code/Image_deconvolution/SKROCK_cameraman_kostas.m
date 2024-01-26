%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%          IMAGE DEBLURRING EXPERIMENT - CAMERAMAN TEST IMAGE            %
%     We implement the SK-ROCK algorithm described in: "Accelerating     %
%    Proximal Markov Chain Monte Carlo by Using an Explicit Stabilized   %
%    Method", Marcelo Pereyra, Luis Vargas Mieles, and Konstantinos C.   %
%    Zygalakis, SIAM Journal on Imaging Sciences, Vol. 13, No. 2, 2020   %
%                Permalink: https://doi.org/10.1137/19M1283719           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

addpath([pwd,'/functions']);

%%% initialize the random number generator to make the results repeatable
rng('default');
%%% initialize the generator using a seed of 1
rng(1);

%% Setup experiment
x=double(imread('cman.png'))/255.0; % Cameraman image to be used for the experiment
x = double(x);
figure(1), imshow(x);
[nRows, nColumns] = size(x); % size of the image


%%%% function handle for uniform blur operator (acts on the image
%%%% coefficients)
h = [1 1 1];
lh = length(h);
h = h/sum(h);
h = [h zeros(1,length(x)-length(h))];
h = cshift(h,-(lh-1)/2);
h = h'*h;


%%% operators A and A'
H_FFT = fft2(h);
HC_FFT = conj(H_FFT);

A = @(x) real(ifft2(H_FFT.*fft2(x))); % A operator
AT = @(x) real(ifft2(HC_FFT.*fft2((x)))); % A transpose operator
ATA = @(x) real(ifft2((HC_FFT.*H_FFT).*fft2((x)))); % AtA operator

% generate 'y'
y = A(x);
BSNRdb = 40; % we will use this noise level
sigma = sqrt(var(y(:)) / 10^(BSNRdb/10));
sigma2 = sigma^2;
y = y + sigma*randn(nRows, nColumns); % we generate the observation 'y'

figure(2), imshow(y)

%%% Algorithm parameters
lambda = sigma2; %%% regularization parameter
alpha = 300.0; %0.044; %%% hyperparameter of the prior
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
logPi = @(x) -(norm(y-A(x),'fro')^2)/(2*sigma2) -0.5*alpha*norm(x,'fro')^2; %%% logpi ||X||^2
%logPi = @(x) -(norm(y-A(x),'fro')^2)/(2*sigma2) -alpha*TVnorm(x); %%% logpi TV
%logPi = @(x) - (norm(y-A(x), 'fro')^2)/(2*sigma2);
%logPi = @(x) - (norm(y-A(x), 'fro')^2)/(2*sigma2) - sum(sum(x/(2*eps^2)*(gmm_fun(x))));


gradient_descent = 0;
sampling = 0;
skrock = 1;
ila = 0;

if gradient_descent
    h=0.9/Lfg;
    N=3000;
    z=y;
    
    energy(1)=logPi(z)
    for i=1:N
        z=z-h*gradU(z);
        energy(i+1)=logPi(z);
    end
    
    figure(3), imshow(z)
    
    
    figure(4), plot([0:1:N],energy)

end

if sampling
    h=0.9*2/Lfg;
    N=1000000;
    z=y;
    meanSamples = z;
    energy(1) = logPi(z)
    mse(1) = immse(meanSamples,x)

    for i = 2:N
        z = z - h*gradU(z) + sqrt(2*h) * randn(nRows, nColumns);
        meanSamples = ((i-1)/i)*meanSamples + (1/i)*z;
        mse(i) = immse(meanSamples,x);
        energy(i) = logPi(meanSamples);
        %energy2(i) = logPi2(meanSamples);

    end

    figure(3), imshow(meanSamples)
    
    
    figure(4), plot([0:1:N-1], energy)
    figure(6), semilogx([0:1:N-1], abs(energy))
    figure(5), plot([0:1:N-1], mse)
    %figure(6), plot([0:1:N-1], energy2)
end

if ila
    h = 0.9*1/Lfg;
    N = 1000;
    z = y;
    meanSamples = z;
    energy(1) = logPi(z)
    mse(1) = immse(meanSamples,x)

    for i = 2:N
        z = ifft(inv((1+h*alpha)*(eye(nRows)) + h/sigma2 * HC_FFT .* H_FFT) * ...
            fft(z + sqrt(2*h) * randn(nRows, nColumns)+ h/sigma2 * ATy));
        meanSamples = ((i-1)/i)*meanSamples + (1/i)*z;
        mse(i) = immse(meanSamples,x);
        energy(i) = logPi(meanSamples);
        %energy2(i) = logPi2(meanSamples);

    end

    figure(3), imshow(meanSamples,[])
    
    
    figure(4), plot([0:1:N-1], energy)
    figure(6), semilogx([0:1:N-1], abs(energy))
    figure(5), plot([0:1:N-1], mse)

end



% ex = ifft2(fft2(AT(y)) ./ (HC_FFT .* H_FFT + 0.07 .* fft2(eye(256))));
% imshow(ex)
% 

if skrock

    % SK-ROCK PARAMETERS
    %%% number of internal stages 's'
    nStagesROCK = 10;
    %%% fraction of the maximum step-size allowed in SK-ROCK (0,1]
    percDeltat = 0.5;
    
    nSamplesBurnIn = 6e2; % number of samples to produce in the burn-in stage
    nSamples = 1e3; % number of samples to produce in the sampling stage
    XkSKROCK = y; % Initial condition
    logPiTrace=zeros(1,nSamplesBurnIn+nSamples);
    logPiTrace(1)=logPi(XkSKROCK);
    %%% to save the mean of the samples from burn-in stage
    meanSamples_fromBurnIn = XkSKROCK;
    %%% to save the evolution of the MSE from burn-in stage
    mse_fromBurnIn=zeros(1,nSamplesBurnIn+nSamples);
    mse_fromBurnIn(1)=immse(meanSamples_fromBurnIn,x);
    %%% to save the mean of the samples in the sampling stage
    meanSamples = zeros(nRows,nColumns);
    %%% to save the evolution of the MSE in the sampling stage
    mse=zeros(1,nSamples);
    
    
    
    %-------------------------------------------------------------------------
    
    disp(' ');
    disp('BEGINNING OF THE SAMPLING');
    
    %-------------------------------------------------------------------------
    
    progressBar = waitbar(0,'Sampling in progress...');
    
    %-------------------------------------------------------------------------
    disp('Burn-in stage...');
    tic;
    for i=2:nSamplesBurnIn
        %%% produce a sample using SK-ROCK
        XkSKROCK=SKROCK(XkSKROCK,Lfg,nStagesROCK,percDeltat,gradU);
        % save \log \pi trace of the new sample
        logPiTrace(i)=logPi(XkSKROCK);
        %%% mean
        meanSamples_fromBurnIn = ((i-1)/i)*meanSamples_fromBurnIn ...
            + (1/i)*(XkSKROCK);
        %%% mse
        mse_fromBurnIn(i) = immse(meanSamples_fromBurnIn,x);
        %%% update iteration progress bar
        waitbar(i/(nSamplesBurnIn+nSamples));
    end
    disp('End of burn-in stage');
    
    disp('Sampling stage...');    
    for i=1:nSamples
        %%% produce a sample using SK-ROCK
        XkSKROCK=SKROCK(XkSKROCK,Lfg,nStagesROCK,percDeltat,gradU);
        % save \log \pi trace of the new sample
        logPiTrace(i+nSamplesBurnIn)=logPi(XkSKROCK);
        %%% mean from burn-in stage
        meanSamples_fromBurnIn = ...
           ((i+nSamplesBurnIn-1)/(i+nSamplesBurnIn))*meanSamples_fromBurnIn ...
           + (1/(i+nSamplesBurnIn))*XkSKROCK;
        %%% mse from burn-in stabe
        mse_fromBurnIn(i+nSamplesBurnIn) = immse(meanSamples_fromBurnIn,x);
        %%% mean from sampling stage
        meanSamples = ((i-1)/i)*meanSamples + (1/i)*XkSKROCK;
        %%% mse from sampling stage
        mse(i) = immse(meanSamples,x);
        %%% update iteration progress bar
        waitbar((i+nSamplesBurnIn)/(nSamplesBurnIn+nSamples));
    end
    
    %-------------------------------------------------------------------------
    
    t_end = toc;
    close(progressBar);
    disp('END OF THE SK-ROCK SAMPLING');
    disp(['Execution time of the SK-ROCK sampling: ' num2str(t_end) ' sec']);
    
    %-------------------------------------------------------------------------%
    % Display MSE associated to the MMSE estimator of x
    disp(['MSE (x): ' num2str(mse(end-1))]);
    %-------------------------------------------------------------------------%
    
    %-------------------------------------------------------------------------%
    % Plot the results                                                        
    %plot_results(y,x,nStagesROCK,meanSamples,logPiTrace,mse);     
    %-------------------------------------------------------------------------%
    save tik_skrock_a300_1e3.mat
end
