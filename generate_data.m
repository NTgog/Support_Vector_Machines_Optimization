function [X, y] = generate_data(N, a, b)
% [X, y] = generate_data(N, a, b)

% Generate n X N matrix X and 1 x N vector y
% separated by hyperplane defined by 
% the n-dimensional vector a and scalar b

% Compute dimension of the data
n = length(a);

% Find a point of the hyperplane
lambda = b / norm(a)^2;
x0 = lambda * a;
mean = a;

for ii=1:N
    y(1,ii) = sign(randn);                             % generate label
    X(:,ii) = x0 + y(1,ii) * mean + .2 * randn(n,1);   % generate data point
end

