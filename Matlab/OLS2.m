function [W] = OLS2(R,C)
% To find the orthonormal basis of X, we use the cross correlation matrix 
% R = X' * X [Nu x Nu]
% To perform OWO, we use the cross-correlation matrix
% C = X' * t [Nu x M]

[Nu] = size(C,1);

eps = 1e-6; eps2 = eps*eps;

A = zeros(Nu, Nu);	% important
% A is the matrix mapping X to iths orthonormal basis X_orth
% i.e., X_orth = X * A'; (but note that A is a lower-traingular matrix)

% (2) For the first basis function, find  a11 as follows.


% The first orthonormal basis is simply X_1 normalized.
% X_orth_1 = X_1 / ||X_1||
% A(1,1) = 1/||X_1|| = 1/sqrt(X_1'*X_1) = 1/sqrt(R(1,1))
% Note: Since the basis function X1 may not always be 1, then r(1,1) may not equal 1.

g = R(1,1);
if(g < eps2)    % check for numerical stability (note: why does it imply a dependent basis?)
    A(1,1) = 0;
else
    g = sqrt(g);
    A(1,1) = 1 / g;
end

% See lecture notes II "Regression in Feedforward Networks", 
% Section G "Conventional Basis Function Analyses",
% subsection 3 "Refining the Schmidt Procedure"
% 
% The n'th orthonormal basis is the n'th basis minus the projections of
% the n'th basis on the previous n-1orthonormal bases
% X_orth_n = X_n - sum( proj(X_n on X_orth_i), for 1<=i<=n-1)
% where,
% proj(X_n on X_orth_i) = <X_n, X_orth_i> * X_orth_i / ||<X_n, X_orth_i>||
% 
% Let c_i = <X_orth_i, X_n>  = A(i,:)*R(:,n) => c ~ A*R(:,n),
% then
% X_orth_n = { X_n - sum( c_i * X_orth_i for 1 <= i <= n-1) } / 
%                    ||X_n - sum( c_i * X_orth_i for 1 <= i <= n-1)||
% 
% X_orth_n = X * b / g
% where
% g = ||X_n - sum( c_i * X_orth_i for 1 <= i <= n-1)|| = R(n,n) - c'*c;
% b(n) = 1, and b(i) ~ -c*A(:,i) for 1<=i<=n-1

% 3) For basis functions 2 through Nu,
for n=2:Nu,
    
    c = A(1:n-1,1:n-1)*R(1:n-1,n);       % [*]
    b = -A(1:n-1,1:n-1)'*c;       % [*]
    b(n) = 1;
    
    g = R(n,n) - c'*c;
    
    if(g < eps2)    % check for numerical stability
        A(n,:) = 0;
    else
        g = sqrt(g);
        A(n,1:n) = b(1:n)'/g;
    end
    
end % end loop over n
% [*] - These can be optimized since A is a sparse (lower traingular)
% matrix, but for simpler notation we have used the full A matrix. For this
% to work correctly, A must be properly initialized to all zeros.

% II-G-4. Uses for Orthonormal Basis Functions
% (4) Find orthonormal system’s output weights as

W_orthonormal = C'*A';

% (5) Find training errors Ei (at each output) and E (total) as

% Ei = Et(:) - sum(W_orthonormal .^ 2, 2);
% E = sum(Ei);

%(6) Find output weights for original system as

W = W_orthonormal*A;

% (7) Print number of linearly dependent basis functions detected
end
