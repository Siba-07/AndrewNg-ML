function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
s = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = sigmoid(X*theta);
J = (-y'*log(z) - (1-y)'*log(1-z))/m + lambda/(2*m)*(sumsq(theta(2:s)));
grad = (X'*(z-y))/m;
k = ones(s);
grad(2:s) = grad(2:s)+ (lambda/m).*theta(2:s);




% =============================================================

end
