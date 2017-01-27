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
%
% 
% The mex files of Liblinear is generated in Windows(64bit).
% If you want to conduct the program in other systems, please compile relevant
% C files of the packages in order to run the program.

%% To repeat the experiments
rng(1);

%% Add necessary pathes
addpath('data','eval');
addpath(genpath('func'));

%% Load the dataset
dataset    = 'enron';
load([dataset,'.mat']);

%% Set global parameters
rate       = 0.2;  % The percentage of labeled instances
dim        = 0.3;  % The percentage of selected features

%% Set parameters of READER
opts.alpha = 1;
opts.beta  = 0.1;
opts.gamma = 10;
opts.k     = 1;
opts.p     = 5;
opts.b     = 1;
opts.maxIt = 50;

%% Perform n-fold cross validation
num_fold = 5;
Results  = zeros(3,num_fold);
indices  = crossvalind('Kfold',size(data,1),num_fold);
for i = 1:num_fold
    rng(i);
    disp(['Round ',num2str(i)]);
    test     = (indices == i); 
    train    = ~test;
    train_id = find(train);
    train_l  = randsample(train_id,round(rate*length(train_id)));
    tic; 
    Fea_Order  = READER(data(train,:),data(train_l,:),target(:,train_l),opts);
    Fea_ID     = Fea_Order(1:round(dim*size(data,2)));
    Pre_Labels = BR(data(train_l,Fea_ID),target(:,train_l),data(test,Fea_ID));
    Results(1,i) = toc; 
    [Results(2:end,i),MetricList] = Evaluation(Pre_Labels,target(:,test));
end
meanResults = squeeze(mean(Results,2));
stdResults  = squeeze(std(Results,0,2) / sqrt(size(Results,2)));

%% Show the experimental results
printmat([meanResults,stdResults],[dataset,'_',num2str(rate),'_',num2str(dim)],['ExeTime ',MetricList],'Mean Std.');
