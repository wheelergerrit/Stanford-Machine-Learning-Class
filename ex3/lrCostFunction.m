function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(X(1,:)); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%The following code is a vectorized way to compute cost for each theta
%cost_vector = y.*log(sigmoid(X*theta)) + (1-y).*log(1-sigmoid(X*theta));
%cost_sum = (1/-m)*sum(cost_vector);

%reg_sum = (lambda/(2*m))*(theta'*theta);

%J = cost_sum + reg_sum

%This next section of code is a vectorized way to compute the gradient
%w.r.t the parameters
%grad = X'*((sigmoid(X*theta)-y)/m); %If there is a problem, its here when 
                                    %computing the vectorized gradient
%temp = theta;
%temp(1) = 0;

%grad = grad + (lambda/m)*temp


hx = sigmoid(X * theta);
lhs = ( -1 .* y .* log(hx));
rhs = ( (1 - y) .* log(1 - hx + eps));

regularize = theta;
regularize(1) = 0;
reg = (regularize' * regularize) * lambda / (2 * m);
J = sum( lhs - rhs)/m + reg;

grad = ( ((hx - y)' * X)'  + (regularize .* lambda)) / m;
% =============================================================

grad = grad(:);

end
