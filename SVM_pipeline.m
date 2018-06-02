%% Import the labeled, unlabeled and test data
data = importfile('train_labeledCSV.csv');
data_un = importfile('train_unlabeledCSV.csv');
data_un = table2array(data_un(:,1:128));
startData = table2array(data(:,1:129));
test = table2array(importfile_test_data('testCSV.csv'));
rest_un = data_un;
added = 501;
currentSet = startData;
%% Start iterating
while size(currentSet,1) ~= 30000
[classifier, acc] = SVM_classifier(currentSet);
[out, score] = classifier.predictFcn(rest_un);
[argvalue, argmax] = max(score');
indices = find(argvalue == 0);
added = size(indices,2)
y = argmax(indices)-1;
currentSet = vertcat(currentSet,horzcat(y',rest_un(indices,:)));
rest_indices = setdiff(linspace(1,size(rest_un,1),size(rest_un,1)),indices);
rest_un = rest_un(rest_indices,:);
acc
end

%% To csc
y = classifier.predictFcn(test);
ready = zeros(8000,2);
ready(:,1) = linspace(30000,37999,8000);
ready(:,2) = y;
csvwrite('sol.csv',ready);