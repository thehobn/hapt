% [e,C,Cn,L] = confumat(K,Y,P) Confusion matrix & classification error
%
% Y(n) contains the ground-truth label (in 1:K) for point X(n). The values of
% X are not necessary. P is a matrix of NxK where row n contains the posterior
% probability p(k|X(n)), for each class k=1:K, for vector X(n). Hence,
% each row contains K values in [0,1] that sum to 1. The predicted label
% has the largest p(k|X(n)).
%
% In:
%   K: number of classes (assumed 1:K), K >= 2.
%   Y: Nx1 list of N ground-truth labels (in 1:K).
%   P: NxK matrix of posterior probabilities.
% Out:
%   e: classification error in [0,1].
%   C: KxK confusion matrix, unnormalised (raw counts).
%   Cn: KxK confusion matrix, normalised (each row in [0,1] with sum 1).
%   L: Nx1 list of N predicted labels (in 1:K).

function [e,C,Cn,L] = confumat(K,Y,P)

    C = zeros(K,K); % KxK matrix
    
    [predVal L] = max(P,[],2);
    for i = 1:size(L,1)
        % C(i,j) is the number of times a class i was classified as class j
        C(Y(i,1),L(i,1)) = C(Y(i,1),L(i,1)) + 1;
    end
    
    Cn = zeros(K,K); % KxK matrix
    rowSums = sum(C,2);
    for i = 1:size(Cn,1)
        for j = 1:size(Cn,2)
           Cn(i,j) = C(i,j) / rowSums(i);
        end
    end

    otherSumC = sum(sum(C - diag(diag(C)))); % sum of all non-diag elements
    e = otherSumC / sum(sum(C));
end










