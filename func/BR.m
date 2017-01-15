function Pre_Labels = BR(train_data,train_target,test_data)
%BR The Binary Relevance [1] Method for MLC 
%
% [1] M. Boutell et al., Learning multi-label scene classification. Pattern Recognition, 2004.

%% Training and testing by the libsvm  
[num_label,num_train]  = size(train_target);
[num_test,num_feature] = size(test_data);

%% Set the SVM package
train_data = sparse(train_data);
test_data  = sparse(test_data);
if num_train > num_feature
    svmlinear = '-s 2 -B 1 -q';
else
    svmlinear = '-s 1 -B 1 -q';
end

%% Build classifiers for each label
Pre_Labels   = zeros(num_test,num_label);     
null_target  = zeros(num_test,1); 
train_target = train_target';
for j=1:num_label
    model           = libtrain(train_target(:,j),train_data,svmlinear);
    Pre_Labels(:,j) = libpredict(null_target,test_data,model,'-q');
end
Pre_Labels = Pre_Labels';

end
