function [grad] = gradient_SVM_barrier(w, X_augm, y, t)
    
% Computes the gradient of gk at the point w
% D gk(w) = tk*w + Ó{yi/(1-yi*X_augm^T*w) * X_augm}

% w is the current point at which we evaluate the gradient
% X_augm is the augmented dataset, with a column of 1's added for the bias term
% y is the vector of labels, t is the barrier parameter

    grad = zeros(size(w));

    for i = 1 : length(X_augm)
        inner_prod = X_augm(:,i).' * w ;
        grad = grad + (y(i)/(1 - y(i)*inner_prod))*X_augm(:,i) ;
    end

    grad = grad + t*w ;
    
end
