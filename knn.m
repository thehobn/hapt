% L = knn(X,Y,x,K) k-nearest neighbour classifier
%
% In:
%   X: NxD matrix containing N D-dimensional data points rowwise (training).
%   Y: Nx1 matrix containing the class labels for X (in 1..K).
%   x: MxD matrix containing M D-dimensional data points rowwise (test).
%   K: (scalar) number of nearest neighbours to use.
% Out:
%   L: Mx1 matrix containing the predicted class labels for x.

function L = knn(X,Y,x,K)
    M = size(x, 1);

    distmat = dist(X,x);

    L = zeros(M, 1);
    for m = 1:M
        [~, I] = sort( distmat(:,m) );
        L(m) = ...
            mode( ...
                Y( ...
                    I(1:K) ...
                ) ...
            );
    end
end

% In:
%   A: N x D N D-dimensional points
%   B: M x D M D-dimensional points
% Out:
%   D: N x M distance:
%       D(N, M) = distance between A(N) and B(M)
function D = dist(A, B)
    N = size(A, 1);
    M = size(B, 1);
    D = zeros(N, M);
    
    for n = 1:N
        D(n, :) = ...
            sqrt( ...
                sum( ...
                    bsxfun( ...
                        @power, ...
                        bsxfun(@minus, A(n, :), B ), ...
                        2 ...
                    ), ...
                    2 ...
                ) ...
            );
    end
end
