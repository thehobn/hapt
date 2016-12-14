% gm = GMclass(X,Y,cov_type) Train Gaussian classifier
%
% Given a labelled training set (X,Y), this computes the maximum likelihood
% estimator for a Gaussian classifier, namely:
% - The prior probabilities (proportion of each class).
% - The mean vectors (average of each class).
% - The covariance matrices (covariance of each class, suitably averaged when
%   the covariance is shared across classes).
% The result is returned in the GMtools format for a Gaussian mixture, where
% each component corresponds to one class.
% See GMpdf.m for descriptions of some of the arguments below.
%
% In:
%   X: NxD matrix containing N D-dim points rowwise.
%   Y: Nx1 matrix containing the class labels for X (in 1..M).
%   cov_type: covariance type (one of 'F','f','D','d','I','i').
% Out:
%   gm: Gaussian mixture struct.

function gm = GMclass(X,Y,cov_type)

classes = unique(Y');

% gm (return value) should be the requsted covarience
gm.type = cov_type;

% find the proportions
proportions = zeros(max(unique(Y)),1);
for i = 1:max(unique(Y))
    proportions(i, 1) = sum(Y(:) == i) / size(Y,1);
end

gm.p = proportions;

% find the mean vectors
meanVectors = [];
for i = 1:max(unique(Y))
    meanVectors = [meanVectors;mean(X(Y==classes(i),:))];
end

gm.c = meanVectors;

% find the covarience stuff
switch cov_type
    case 'F'
        covs = zeros(size(X,2), size(X,2), max(unique(Y))); % DxDxN matrix
        for i = 1:max(unique(Y)) % loop through every class
            temp = X(Y==i,:); % all X's for that class
            covs(:,:,i) = cov(temp); % store the covarience of the class
        end
        
        gm.S = covs;
    case 'f'
        covSum = zeros(size(X,2), size(X,2)); % DxD matrix
        for i = 1:max(unique(Y)) % loop through every class
            temp = X(Y==i,:); % all X's for that class
            covSum = covSum + proportions(i,1) * cov(temp);
        end
        
        gm.S = covSum;
    case 'D'
        diags = zeros(max(unique(Y)), size(X,2)); % NxD matrix
        for i = 1:max(unique(Y)) % loop through every class
            temp = X(Y==i,:); % all X's for that class
            tempCov = cov(temp); % covarience of the class
            diags(i,:) = diag(tempCov);
        end
        
        gm.S = diags;
    case 'd'
        diagSum = []; % 1xD matrix
        for i = 1:max(unique(Y)) % loop through every class
            temp = X(Y==i,:); % all X's for that class
            tempCov = cov(temp); % covarience of the class            
            diagSum = [diagSum; proportions(i,1) * diag(tempCov)'];
        end
        
        gm.S = sum(diagSum);
    case 'I'
        sigmas = zeros(max(unique(Y)), 1); % Nx1 matrix
        for i = 1:max(unique(Y))
            temp = X(Y==i,:); % all X's for that class
            tempCov = cov(temp); % covarience of the class
            sigmas(i,1) = sum(diag(tempCov))/size(diag(tempCov),1);
        end
        
        gm.S = sigmas;
    case 'i'
        sigmaSum = zeros(1, 1); % 1x1 matrix
        for i = 1:max(unique(Y))
            temp = X(Y==i,:); % all X's for that class
            tempCov = cov(temp); % covarience of the class
            scaledDiag = proportions(i,1) * diag(tempCov);
            sigmaSum(1,1) = sigmaSum(1,1) + sum(scaledDiag)/size(scaledDiag,1);
        end
        
        gm.S = sigmaSum;
    otherwise
        spy
end













