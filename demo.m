% This is an example program for the paper: 
% 
% L. Sun, M. Kudo and K. Kimura, READER: Robust Semi-Supervised Multi-Label Dimension Reduction. 
% A submission to the IEICE Trans. on Information and Systems. 
%
% The program shows how the READER program (The main function is 'READER.m') can be used.
%
% Please type 'help READER' under MATLAB prompt for more information.
%
% The program was developed based on the following packages and codes:
%
% [1] Liblinear
% URL: https://www.csie.ntu.edu.tw/~cjlin/liblinear/
%
% [2] Matlab codes for dimensionality reduction
% URL: http://www.cad.zju.edu.cn/home/dengcai/Data/DimensionReduction.html

%% To repeat the experiments
rng('default');

%% Add necessary pathes
addpath('data','eval');
addpath(genpath('func'));

%% Load a multi-label dataset and method
dataset    = 'enron';
method     = 'alg2';   % alg1, alg2
load([dataset,'.mat']);

%% Set global parameters
rate       = 0.3;  % The percentage of labeled instances
dim        = 0.3;  % The percentage of selected features

%% Set parameters of READER
opts.alpha = 1;
opts.beta  = 0.1;
opts.gamma = 10;
opts.k     = 0.1;
opts.p     = 5;
opts.maxIt = 100;
opts.epsIt = 1e-3;

%% Perform n-fold cross validation
numFold = 5;
Results = zeros(3,numFold);
indices = crossvalind('Kfold',size(data,1),numFold);
numS = round(dim*size(data,2));
for i = 1:numFold
    disp(['Round ',num2str(i)]);
    test  = (indices == i); 
    train = ~test;
    idXl  = randsample(find(train),round(rate*length(find(train))));
    tic;
    switch method
        case 'alg1'
            idF = READERalg1(data(train,:),data(idXl,:),target(:,idXl),opts);
        case 'alg2'
            idF = READER(data(train,:),data(idXl,:),target(:,idXl),opts);
    end
    idS = idF(1:numS);
    Yt  = BR(data(idXl,idS),target(:,idXl),data(test,idS));
    Results(1,i) = toc; 
    [Results(2:end,i),MetricList] = Evaluation(Yt,target(:,test));
end
meanResults = squeeze(mean(Results,2));
stdResults  = squeeze(std(Results,0,2) / sqrt(size(Results,2)));

%% Show the experimental results
printmat([meanResults,stdResults],[dataset,'_',method,'_',num2str(rate),'_',num2str(dim)],['ExeTime ',MetricList],'Mean Std.');