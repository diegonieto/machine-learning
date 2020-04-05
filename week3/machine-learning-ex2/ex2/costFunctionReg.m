function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

part1 = -y.*log(sigmoid(X*theta));       % y == 1 => low cost
part2 = (1-y).*log(1-sigmoid(X*theta)); % y == 0 => high cost

thetaReg = theta
thetaReg(1) = 0 % Do not regularize theta zero
regularization = sum(thetaReg.^2)*lambda/(2*m)

J = sum(part1-part2)/m + regularization;

grad = 1/m * X'*(sigmoid(X*theta)-y) + (thetaReg*lambda/m)

end
