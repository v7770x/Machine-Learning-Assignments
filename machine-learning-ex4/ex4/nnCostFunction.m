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
a2 = (sigmoid(Theta1*[ones(m,1) X]'))';
a3 = sigmoid((Theta2*[ones(1,m);a2'])');
yFull = zeros(m,num_labels);
for i = 1:m,
  yFull(i, y(i,1))=1;
end;
J = -1/m * sum(sum(yFull.*log(a3)+(1-yFull).*
  log(1-a3)));

regTerm = lambda/(2*m) * sum([(Theta1(:, 2:end).^2)(:);(Theta2(:,2:end).^2)(:)]);
J = J+regTerm;


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

X = [ones(m,1) X]; %mX401
z2 = Theta1*X'; %25*401 X 401*m
size(z2)
a2 = [ones(1,m) ; sigmoid(z2)]; %26Xm
z3 = Theta2*a2; %10*26 X 26*m
a3 = sigmoid(z3); %10Xm
s3 = a3 - yFull'; %10Xm 

s2 = (s3'*Theta2).* (sigmoidGradient([ones(1,m); z2]))'; %m*26
s2 = s2(:,2:end);


Theta1_grad = 1/m *(s2'*X); %25Xm X mX401
Theta2_grad = 1/m * (s3*a2'); %10Xm X mX26

%{
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(m,1) X];
for i = 1:m,
  currX = (X(i , :))'; %401X1
  currY = (yFull(i,:))';%10X1
  cz2 = [Theta1*currX]; %25X1
  ca2 = [1; sigmoid(cz2)]; %26X1
  cz3 = Theta2*ca2; %10X1
  ca3 = sigmoid(cz3); %10X1
  cS3 = ca3-currY; %10X1
  cS2 = Theta2'*cS3.*sigmoidGradient([1;cz2]); cS2 = cS2(2:end);%size = 25X1
  Theta1_grad = Theta1_grad + cS2*currX';
  Theta2_grad = Theta2_grad + cS3*ca2';
end

Theta1_grad = 1/m*Theta1_grad;
Theta2_grad = 1/m *Theta2_grad;
  
%}



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:, 2:end);






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
