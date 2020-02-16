function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint:theta(1), theta(2) While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    %fprintf('Cost: %f\n', J_history(iter));
    
    % vectorized way
    %theta = theta - (alpha/m * X'*(X*theta - y));
    
    % non vectorized way
   % J = 0;
   % for j = 1:m
   %   tmp = zeros(m,1)
   %   for i = 1:m
   %     h = theta(1) + theta(2)*X(i,2);
   %     tmp(i) = h-y(i)
   %   end
   %   theta_1_tmp = X(i,2)*tmp(i)
   % end
   % theta(1) = theta(1) - (alpha/m * theta_1_tmp);
   % theta(2) = theta(2) - (alpha/m * theta_1_tmp);
    
    
        tmp_j1=0;
for i=1:m, 
    tmp_j1 = tmp_j1+ ((theta (1,1) + theta (2,1)*X(i,2)) - y(i));
end

    tmp_j2=0;
for i=1:m, 
    tmp_j2 = tmp_j2+ (((theta (1,1) + theta (2,1)*X(i,2)) - y(i)) *X(i,2)); 
end

    tmp1= theta(1,1) - (alpha *  ((1/m) * tmp_j1))  
    tmp2= theta(2,1) - (alpha *  ((1/m) * tmp_j2))  

    theta(1,1)=tmp1
    theta(2,1)=tmp2

end

end
