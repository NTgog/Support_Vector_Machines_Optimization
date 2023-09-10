function [ Hess ] = Hess_SVM_barrier( w, X_augm, y, t)

% Computes the Hessian of gk at the point w
% D^2 gk = tk*I + Ó{(1/(yi*Xi^T*w-1)^2)*Xi*Xi^T}

% w is the current point at which to evaluate the Hessian
% X_augm is the augmented dataset, with a column of 1's added for the bias term
% y is the vector of labels, t is the barrier parameter

    Hess = zeros(size(w,1),size(w,1)) ;
    
    for i = 1 : length(X_augm)
        inner_prod = X_augm(:,i).'*w ;
        Hess = Hess + (1/((-1 + y(i)*inner_prod)^2))*(X_augm(:,i)*X_augm(:,i).') ;
    end
    
    Hess = Hess + t*eye(size(Hess)) ;

end
    

