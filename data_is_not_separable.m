function flag = data_is_not_separable(X, y, a, b)
% a = data_is_not_separable(X, y, a, b)

N = length(y);
flag = 0;

tmp = y' .* (X' * [b; a]) > 0;
if ( sum(tmp) < N )
    flag = 1;
end