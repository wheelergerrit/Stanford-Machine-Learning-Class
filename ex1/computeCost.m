function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
i = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
for n = 1:i
    J = J + ((theta'*X(n,:)')-y(n,:))^2
end
J = J*(1/(2*i))
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.




% =========================================================================

end
