% [f,e,sv,l,m,xi] = linsvmtrain(X,Y,C,dual) Train binary linear SVM
%
% This trains a linear SVM for binary classification. There are two cases:
% - Linearly separable case (C=NaN): optimal separating hyperplane, if one
%   exists (check the error code e to see whether the QP was feasible). It
%   doesn't return slack variables.
% - Not linearly separating case (0 < C < Inf): soft margin hyperplane.
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   Y: Nx1 vector, N labels in {-1,+1}.
%   C: nonnegative scalar, penalty parameter for the soft margin hyperplane.
%      C = NaN means the optimal separating hyperplane.
%   dual: 0 to solve the primal, 1 to solve the dual.
% Out:
%   f: (struct) the SVM.
%   e: quadprog's error code:
%      1 (KKT point), 0 (maxit exceeded), -2 (infeasible), -3 (unbounded), etc.
%   sv: list of indices in 1:N of the support vectors in X.
%   l: Nx1 vector, the Lagrange multiplier for each data point.
%   m: margin=1/|f.w|, distance from the hyperplane to its closest point in X.
%   xi: Nx1 vector, slack variable (constraint violation) for each data point
%      (not returned for the optimal separating hyperplane).

function [f,e,sv,lam,m,xi] = linsvmtrain(X,Y,C,dual)

[N,L] = size(X);
X1 = [X,ones(N,1)];
Y1 = -Y;
A = bsxfun(@times,Y1,X1);
b = -ones(N,1);
if (isnan(C))
    H = eye(L+1);
    [l,w] = size(H);
    H(l,:) = 0;
    F = zeros(L+1,1);
    [x,fval,e,output,lam]= quadprog(H,F,A,b);%black magic
    [sv,j] = find((Y.*(X1*x))-1<1e-5);
    lam = lam.ineqlin;
    [p,o]=size(x);
    f.w = x(1:p-1); 
    f.w0=x(p);
    m=1/norm(f.w);
    xi=0;
    
else
    H = diag([ones(1,L) 0 zeros(1,N)]);
    A = [A -eye(N)];
    F = C*[zeros(L+1,1); ones(N,1)];
    lu = [-Inf*ones(L+1,1);zeros(N,1)]; 
    Aeq = zeros(1,L+N+1);
    [x,fval,e,output,lam] = quadprog(H,F,A,b,Aeq,0,lu,Inf);%black magic
    [p,o] = size(x);
    f.w = x(1:L); 
    f.w0 = x(L+1); 
    m = 1/norm(f.w);
    xi = x(L+2:p);
    [sv,j] = find((Y.*(X1*x(1:L+1))) - 1 < 0);
    lam = lam.ineqlin;
end