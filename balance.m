myTarget = zeros(10,30000);
for i=1:30000
vec = zeros(10,1);
vec(labTargets(i)+1) = 1;
myTarget(:,i) = vec;
end

%data = table2array(importfile('train_labeledCSV.csv'));
data = extendeddata;
y = data(:,1);
vec = zeros(10,1);
for i=1:10
vec(i) = numel(find(y==i-1));
end
minn = min(vec);
indices = zeros(100,1);
for i=1:10
    if vec(i) ~= minn
        diff = vec(i) - minn;
        indices = vertcat(indices,find(y==i-1,diff));
    end
end
keep = setdiff(linspace(1,30000,30000),indices);
new_data = data(keep,:);
