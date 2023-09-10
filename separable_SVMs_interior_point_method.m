%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% M-file that implements an interior-point algorithm for the           %
% solution of a classification problem using SVMs - we assume          %
%                     separable data                                   %
%                                                                      %
% Objective function: f_0(w) = 1/2 * || w||^2                          %
%                                                                      %
% Inequality constraints:                                              %
%            f_i(w) = 1 - y_i w^T \bar_x_i \le 0, i=1,...,n            %
%                                                                      %
% Method of solution: Barrier                                          %
%                                                                      %
% A. P. Liavas, Dec. 3, 2022                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc, clear, figure(1), clf, %close all

n = 2;        % problem dimension
N = 1000;     % number of data points

% Define and draw the separating line a^T * x = b
a = randn(2,1); b = randn;
x1 = [-2:.1:2];
for ii=1:length(x1)
    x2(ii) = 1/a(2) * ( -a(1) * x1(ii) + b );
end
plot(x1, x2, '-'), grid, hold on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate and draw SEPARABLE data to classify
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nData generation and plot...')
[X, y] = generate_data(N, a, b);
for ii=1:N
    if (y(ii) == 1)
       plot(X(1,ii), X(2,ii), 'bo')
    else
        plot(X(1,ii), X(2,ii), 'r+')
    end
end
% Augment the data matrix with -1 elements
X_augm = [-ones(1,N); X];
% Check data separablity with respect to the true a and b 
if ( data_is_not_separable(X_augm, y, a, b) == 0 ), 
    fprintf('\nData is separable... I proceed to the solution...')
    pause(1)
else
    fprintf('\nData is non-separable... I quit...')
    break
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution via SVMs and CVX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nSolution of SVM via CVX (primal problem)...')
cvx_begin quiet
   variable w_SVM(n+1, 1);
   minimize ( w_SVM' * w_SVM );
        subject to 
               y' .* (X_augm' * w_SVM) >= 1;
cvx_end

% Draw SVM-based separating hyperplane (color: magenta) 
% in the same plot with "true" separating hyperplane and data
a_est = w_SVM(2:3);
b_est = w_SVM(1);
x1 = [-2:.1:2];
for ii=1:length(x1)
    x2(ii) = 1/a_est(2) * ( -a_est(1) * x1(ii) + b_est );
end
figure(1), plot(x1, x2, 'm'), grid on, 
xlabel('x1'), ylabel('x2'), hold on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution via interior point method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nSolution via interior point method...')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization 
w_init(:,1) = [b; a];  % start from a feasible point
while ( point_is_feasible(w_init, X_augm, y) == 0 )
     w_init = 2 * w_init;
end
% *******************************************************

% % Initialization 
threshold_1 = 10^(-3);                % determines termination condition for inner Newton iterations
epsilon = 10^(-5);                    % determines final termination condition and algorithm accuracy 
alpha = 0.3; beta = 0.7;              % backtracking parameters
t = 10;                               % initialize parameter t of barrier method
mu = 10;                              % used for the increase of t: t = mu * t;

outer_iter = 1;
W(:,outer_iter) = w_init;

while (1)                                     % Outer loop 
      
       w(:,1) = W(:, outer_iter);    % Start new optimization from the previous solution
       inner_iter = 1;               % initialize counter of inner iterations
       
       while ( 1 )                    % Inner loop 
          
          grad = gradient_SVM_barrier( w(:,inner_iter), X_augm, y, t );      % gradient at w
          Hess = Hess_SVM_barrier( w(:,inner_iter), X_augm, y, t );          % Hessian at w
          Dw_Nt =  - pinv(Hess) * grad;                                      % Newton step
          l_x = sqrt( Dw_Nt' * Hess * Dw_Nt );                               % Newton decrement
          if (l_x^2/2 <= threshold_1) break; end                             % Newton termination condition

           % Update and check feasibility - if the new point is not feasible, then backtrack 
           tau = 1; w_new = w(:,inner_iter) + tau * Dw_Nt;
           while (  point_is_feasible(w_new, X_augm, y) == 0  ) 
                fprintf('\nNew point is infeasible... I backtrack...')
                tau = beta * tau;
                w_new = w(:,inner_iter) + tau * Dw_Nt;
           end
           
           % Backtracking line search 
           while ( barrier_SVM_cost_function( w_new, X_augm, y, t ) > ...
                            barrier_SVM_cost_function( w(:,inner_iter), X_augm, y, t ) + alpha * tau * grad' * Dw_Nt )
                fprintf('\nI backtrack...')        
                tau = beta * tau; 
                w_new = w(:,inner_iter) + tau * Dw_Nt;
           end
 
           % Update and plot w in the same plot with w_SVM
           w(:,inner_iter+1) = w_new;
           
           % Plot current solution estimate and optimal solution computed via CVX
           figure(2), plot([w_SVM w(:,inner_iter+1)]), 
           legend('optimal solution', 'current estimate'), pause(.001)     
           
           inner_iter = inner_iter + 1;
       end
 
       W(:, outer_iter+1) = w(:, inner_iter);           % W(:, outer_iter+1): solution of optimization problem      
       if (N/t < epsilon) break; end                % Algorithm termination condition
       outer_iter = outer_iter + 1;
       t = t * mu;
end