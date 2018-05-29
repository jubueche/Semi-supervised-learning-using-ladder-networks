%% Load data
%{
data = table2array(importfile('train_labeledCSV.csv'));
data_unlabeled = table2array(importfile('train_unlabeledCSV.csv'));
test_data = table2array(importfile_test_data('testCSV.csv'));
X = data(:,2:129);
y = data(:,1);  
%}
%% Load more features
data = load('more_features.mat');
data = data.extended;
X = data(:,2:end);
y = data(:,1);
best = 0;
predictors = X(:,1);
%% Train the classifier
response = y;
for i=2:size(X,2)
rng(1)
tmp_pred = horzcat(predictors,X(:,i));
template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    tmp_pred, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsall', ...
    'ClassNames', [0; 1; 2; 3; 4; 5; 6; 7; 8; 9]);

%% Add things to the struct
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(x);

trainedClassifier.ClassificationSVM = classificationSVM;

%% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError')
if validationAccuracy >= best
    %add to the predictors
    i
    predictors = horzcat(predictors,X(:,i));
    best = validationAccuracy;
end
end
%% Write to csv
predicted_test_data = trainedClassifier.predictFcn(test_data);

ready_csv = zeros(8000,2);
ready_csv(:,1) = linspace(30000,37999,8000);
ready_csv(:,2) = predicted_test_data;

csvwrite('sol.csv',ready_csv);

%% Predict unlabled data
predicted_unlabeled = trainedClassifier.predictFcn(data_unlabeled(:,1:128));
extended = zeros(30000,129);
extended(1:9000,:) = data;
extended(9001:end,1) = predicted_unlabeled;
extended(9001:end,2:129) = data_unlabeled(:,1:128);

csvwrite('extended_data.csv',extended);