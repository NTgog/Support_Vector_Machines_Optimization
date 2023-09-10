function [gk] = barrier_SVM_cost_function( w_new, X_augm, y, t)

% Computes the value of gk at the point w_new : gk(w) = t * f0(w) + ö(w)
% where f0(w) = 0.5 * ||w_new||_2^2, ö(w) = -Ó{log(-fi(w))}

% w_new is the current point at which to evaluate the cost function
% X_augm is the augmented dataset, with a column of 1's added for the bias term
% y is the vector of labels, t is the barrier parameter
    
    gk = 0 ;
    
    for i = 1 : length(X_augm)
        inner_prod = w_new.'*X_augm(:,i) ;
        gk = gk + log(-(1 - y(i)*inner_prod)) ;
    end
    
    gk = - gk + (t/2)*norm(w_new,2)^2 ;

end
