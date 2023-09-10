function flag = point_is_feasible(w, X_augm, y)

% If w_init \in dom÷ flag = 1, otherwise flag = 0
% dom÷ = {w \in R^n | fi(w) < 0}, where fi(w) = 1 - yi*w^T*X_augm_i

% w_init : The initial point to check for feasibility
% X_augm : The augmented dataset 
% y : the vector of labels
   
    fi = ones(size(y)) - y.*(w'*X_augm);
    max_fi = max(fi);
    
    flag = 1;
    if max_fi >= 0 
        flag = 0; % If fi>=0 for at least one data point 
    end

end