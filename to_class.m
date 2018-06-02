function [y] = to_class(X)
for i=1:size(X,2)
    [argval, argmax] = max(X(:,i));
    y(i) = argmax-1;
end
end

