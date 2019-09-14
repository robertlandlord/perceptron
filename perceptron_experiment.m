function [num_iters, bounds_minus_ni] = perceptron_experiment(N, d, num_samples)
% perceptron_experiment: Code for running the perceptron experiment in HW1
% Inputs:  N is the number of training examples
%          d is the dimensionality of each example (before adding the 1)
%          num_samples is the number of times to repeat the experiment
% Outputs: num_iters is the # of iterations PLA takes for each sample
%          bound_minus_ni is the difference between the theoretical bound
%                         and the actual number of iterations
%          (both the outputs should be num_samples long)
import perceptron_learn.*
    counter = 0;
    num_iters = zeros(num_samples,1);
    bounds_minus_ni = zeros(num_samples,1);
while counter<num_samples
    w_star = rand(1,d+1);
    w_star(1,1) = 0;
    training_set = vertcat(ones(1,N), -1 + (1+1)*rand(d,N));
    y = sign(w_star*training_set);
    %calculating the theoretical bound t \leq R^2||w*||^2/rho^2
    %must be done before we join the x and the y together
    rho = min(y.*(w_star*training_set));
    R = max(vecnorm(training_set));
    theoretical_bound = (R^2)*((norm(w_star))^2)/(rho^2);
    %now we can append the training set of x's and the y together
    training_set = vertcat(training_set,y);
    [w,iterations] = perceptron_learn(training_set);
    
    %Check to make sure it worked correctly -- this part also keeps track
    %of iterations
    y_check = zeros(1,N);
    for i = 1:1:N
        y_check(1,i) = dot(transpose(w),training_set(1:d+1,i));
    end 

    isCorrect = true;
    for j = 1:1:N
        if sign(y(1,j)) ~= sign(y_check(1,j))
            isCorrect = false;
        end
    end
    
    if(~isCorrect)
        disp("Ran into a misclassified number!")
    else %If everything ran correctly, we need to calculate the theoretical bound now
        num_iters(counter+1,1) = iterations;
        bounds_minus_ni(counter+1,1) = log(theoretical_bound - iterations);
    end
    counter = counter + 1;
end

end
