%% Import the labeled, unlabeled and test data
data = importfile('train_labeledCSV.csv');
data_un = importfile('train_unlabeledCSV.csv');
data_un = table2array(data_un(:,1:128));
startData = table2array(data(:,2:129));
startTarget = to_nn_output(table2array(data(:,1))',10);

%% Predict and select run1
out = nn(data_un);
[argvalue, argmax] = max(out');
indices = find(argvalue > 0.999);
argmax = argmax(indices);

run1Data = vertcat(startData,data_un(indices,:));
run1Target = zeros(10,size(indices,2));
for i=1:size(indices,2)
    run1Target(argmax(i),i) = 1;
end
run1Target = horzcat(startTarget,run1Target)';

%% Predict and select run2
rest1 = setdiff(data_un,data_un(indices,:));
out = nn1(data_un);
[argvalue, argmax] = max(out');
indices = find(argvalue > 0.999);
argmax = argmax(indices);

run2Data = vertcat(run1Data,data_un(indices,:));
run2Target = zeros(10,size(indices,2));
for i=1:size(indices,2)
    run2Target(argmax(i),i) = 1;
end
run2Target = horzcat(run1Target,run2Target)';