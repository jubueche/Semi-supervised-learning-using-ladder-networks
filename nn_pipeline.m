%% Import the labeled, unlabeled and test data

data = importfile('train_labeledCSV.csv');
X_un = importfile('train_unlabeledCSV.csv');
X_un = table2array(X_un(:,1:128))';
X = table2array(data(:,2:129))';
y = table2array(data(:,1));
test = table2array(importfile_test_data('testCSV.csv'))';

%% Import standardized data
X_un = importstand('standard_train_unlabeled.csv');
X = importstand('standard_train_labeled.csv');
X_un = X_un(:,1:128)';
X = X(:,1:128)';

%% Import Kernel PCA data
%{
X_un = import_pca('pca_train_unlabeled.csv')';
X = import_pca('pca_train_labeled.csv')';
test = import_pca('pca_test.csv')';

%% Import 32 Kernel PCA
X_un = import_pca_32('pca_train_unlabeled_32.csv')';
X = import_pca_32('pca_train_labeled_32.csv')';
test = import_pca_32('pca_test_32.csv')';

%% Normal PCA

numCom = 100;
[COEFF_x, X] = pca(X','NumComponents',numCom);
[COEFF_x_un, X_un] = pca(X_un','NumComponents',numCom);
[COEFF_t, test] = pca(test','NumComponents',numCom);
X = X';
X_un = X_un';
test = test';
%}
%% Set Hyperparams
currentX = X;
currentY = y;
rest_un = X_un;

minSize = 100;
maxSize = 100;
numLayers = 1;
regularization = 0;
added = 100;
threshold = 0.99;
kFold = 5;
prev_acc = 0;
%% Train NN

while added >=100
    performanceVec = zeros(5,1);
    targets = to_nn_output(currentY',10);
    best = 0;
    for s=minSize:100:maxSize
        % Create nn with specified parameters
        cross_val = zeros(kFold,1);
        for k=1:kFold
            k
            net = patternnet(s*ones(1,numLayers));
            net.divideFcn = 'divideind';
            upper = round(size(currentX,2)*0.3);
            net.divideParam.trainInd = upper+1:size(currentX,2);
            lin = randperm(upper);
            net.divideParam.valInd = lin(1:round(upper/2));
            net.divideParam.testInd  = lin(round(upper/2)+1:upper);
            net.performParam.regularization = regularization;
            rng(1); % For reprod.
            net = train(net,currentX,targets);
            outputs = net(currentX);
            performanceVec(s) = perform(net,targets,outputs);
            % Calculate the accuracy on the test set
            out_test = net(currentX(:,1351:2700));
            [~, argmax] = max(out_test);
            out_test = argmax-1;
            actual_test = vec2ind(targets(:,1351:2700))-1;
            accuracy = sum(actual_test == out_test)/size(actual_test,2);
            cross_val(k) = accuracy;
        end
        if mean(cross_val) > best
            best_net = net;
            best = mean(cross_val)
        end
    end
    prev_acc = best
    % best_net holds the net with the best performance
    out = best_net(rest_un);
    [argvalue, argmax] = max(out);
    indices = find(argvalue > threshold);
    added = numel(indices)
    size(currentX)
    y = argmax(indices)-1;
    currentX = horzcat(currentX,rest_un(:,indices));
    currentY = vertcat(currentY,y');
    rest_indices = setdiff(linspace(1,size(rest_un,2),size(rest_un,2)),indices);
    rest_un = rest_un(:,rest_indices);
end

%% To csv
out = best_net(test);
y = to_class(out);
ready = zeros(8000,2);
ready(:,1) = linspace(30000,37999,8000);
ready(:,2) = y;
csvwrite('sol.csv',ready);