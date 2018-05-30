function [X] = to_nn_output(y,n)
% N is the number of classes
X = zeros(n,size(y,2));
for i=1:size(y,2)
    X(y(i)+1,i) = 1;
end
end

