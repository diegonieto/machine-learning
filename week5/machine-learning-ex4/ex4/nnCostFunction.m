function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1 -> Feedforward propagation

% 3 layers
% Layer 1 -> 20x20 pixeles + bias = 400+1
% Layer 2 -> 25 units + bias
% Layer 3 -> 10 outsputs

% X = NumberOfExamples X Picture pixels   -> 5000 x 400
% Y = NumberOfExamples X Value [0-9 digit]-> 5000 x 1
% Theta 1 -> Layer 1 of weights 25x401
% Theta 2 -> Layer 1 of weights 10x26

J = 0;

% Adding bias to input layer
input_layer_bias = [ones(size(X),1) X]; % -> 5000x401
sigmoid_input_layer = sigmoid(input_layer_bias*Theta1');    % 5000x401 * 401*25 -> 5000x25

% Adding bias to hidden layer
hidden_layer_bias = [ones(size(X),1) sigmoid_input_layer];  % 5000x25+BIAS COLUMN -> 5000x26
sigmoid_hidden_layer = sigmoid(hidden_layer_bias*Theta2');  % 5000x26 * 26x10 -> 5000x10

% Compare real output with result -> 5000x10
% For each example we generate a vector setting to 1 only the index
% of the associated digit
yVector = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

% Compute partial cost
part1 = -yVector.*log(sigmoid_hidden_layer);       % y == 1 => low cost
part2 = (1-yVector).*log(1-sigmoid_hidden_layer);  % y == 0 => high cost

% Regularize without applying bias column in (1) for Thetas
regularization = (sum(sum(Theta1(:, 2:end).^2))+sum(sum(Theta2(:, 2:end).^2)))*lambda/(2*m);

J = sum(sum(part1-part2))/m + regularization;

% -------------------------------------------------------------


% Part 2 implement back propagation

% propagation formula:
% (L)delta_unit = (L)a - (i)y
% (l)Delta(i)(j) = (l)Delta(i)(j) + (l)a(j)*(l+1)delta_unit(i)

% Initialize delta to zeros
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

% Walk through the examples applying deltas
for example_iter = 1:m,
	a1t_weights = input_layer_bias(example_iter,:)';
	a2t_weights = hidden_layer_bias(example_iter,:)';
  
 	sigt = sigmoid_hidden_layer(example_iter,:)';
	yVectorT = yVector(example_iter,:)';
  % Calculate delta unit (L)delta_unit = (L)a - (i)y
	d3t = sigt - yVectorT;

	z2t = [1; Theta1 * a1t_weights];
  % Calculate delta unit (L)delta_unit = (L)a - (i)y
	d2t = Theta2' * d3t .* sigmoidGradient(z2t);

  % (l)Delta(i)(j) = (l)Delta(i)(j) + (l)a(j)*(l+1)delta_unit(i)
	delta1 = delta1 + d2t(2:end) * a1t_weights';
	delta2 = delta2 + d3t * a2t_weights';
end;

% Calculate deltas for theta
Theta1ZeroBias = [ zeros(size(Theta1, 1), 1) Theta1(:, 2:end) ];
Theta2ZeroBias = [ zeros(size(Theta2, 1), 1) Theta2(:, 2:end) ];
Theta1_grad = (1 / m) * delta1 + (lambda / m) * Theta1ZeroBias;
Theta2_grad = (1 / m) * delta2 + (lambda / m) * Theta2ZeroBias;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
