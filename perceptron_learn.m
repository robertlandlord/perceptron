function [w, iterations] = perceptron_learn(data_in)
% perceptron_learn: Run PLA on the input data
% Inputs:  data_in is a matrix with each row representing an (x,y) pair;
%                 the x vector is augmented with a leading 1,
%                 the label, y, is in the last column
% Outputs: w is the learned weight vector; 
%            it should linearly separate the data if it is linearly separable
%          iterations is the number of iterations the algorithm ran for
dimensionality = size(data_in,1)-2; % we subtract 2 to remove the leading 1 and the y row (we don't count the augment as a dimension)
w = zeros(1,dimensionality+1); %added one to account for the augmented dimension
iterations = 0;
hasMisclassified = true;
numColumns = size(data_in,2);
while(hasMisclassified)
    for i = 1:1:numColumns
        x = data_in(1:dimensionality+1,i); %gets all x's including the augmented row
        y = data_in(dimensionality+2,i); %retrieves the last row of matrix
        if sign(w*x) ~= sign(y)% if its misclassified, then reweight
            w = w + transpose(x)*y;
        end
    end
    hasMisclassified = false;
    for j = 1:1:numColumns
        x = data_in(1:dimensionality+1,j);
        y = data_in(dimensionality+2,j);
        if sign(w*x) ~= sign(y)
            hasMisclassified = true;
        end
    end
    iterations = iterations + 1;
end
