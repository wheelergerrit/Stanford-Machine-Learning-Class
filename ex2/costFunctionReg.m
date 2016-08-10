function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(X(1,:)) %number of features in training matrix

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%The chunk of code below computes the regularized cost function for theta
cost_sum = 0
for i = 1:m
    new_cost = (sigmoid(X(i,:)*theta)-y(i))^2
    cost_sum = cost_sum + new_cost
end
reg_sum = 0
for j = 1:n
    reg_term = theta(j)^2
    reg_sum = reg_sum + reg_term
end

 J = (cost_sum + lambda*reg_sum)/(-2*m)

%This next section of code computes the regularized gradient for each
%particular theta value

%The gradient for theta(1) must be calculated separately as it does not get
%regularized
grad_sum = 0
for i = 1:m
   grad_sum = grad_sum + (sigmoid(X(i,:)*theta)-y(i))*X(i,1)
end
grad(1) = grad_sum/m

for j = 2:length(theta)
    grad_sum = 0
    for i = 1:m
       grad_sum = grad_sum + (sigmoid(X(i,:)*theta)-y(i))*X(i,j)
    end
    grad(j) = (grad_sum + lambda*theta(j))/m
end



% =============================================================

end
