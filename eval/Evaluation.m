function [Results,MetricList] = Evaluation(Pre_Labels,test_target)
%EVALUATION Multi-label Evaluation
%
%      Syntax:
% 
%          [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,test_target)
%
%      Input:
%
%          Pre_Labels          L x Nt predicted label matrix           
%          test_target         L x Nt groundtruth label matrix
%
%      Output:
%
%          MacroF1             Macro-averaged F1 measure
%          MicroF1             Micro-averaged F1 measure

    %% Save evalutation results
    Results = zeros(2,1);
    Results(1,1) = Macro_F1(test_target,Pre_Labels);  
    Results(2,1) = Micro_F1(test_target,Pre_Labels);
    
    %% Save the list of used metrics
    MetricList = 'MacroF1 MicroF1';

end
