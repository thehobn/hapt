load hapt.mat

X = hapt.train.feature_data;
Y = hapt.train.activity_labels;

% Subset of points to compute LDA on
I = find(Y<15);	% [1,2,3,4]
%I = datasample(I,1000,'Replace',false);	% rnd subset

% Instances X is of DxN, labels Y is of 1xN, K = number of classes
X = X(I,:); Y = Y(I);

% 1x1 L number of dimensions to have in subspace
L = 11;

[~,Sw,Mk,Sb,U,d,J] = lda(X,Y);
[W,lsSw,lsSb,Z] = lda_project(U,J,L,Sb,Sw,X);

%{
    1. Eigenvalues in d, L in L
    2. Means of each class in Mk
    3. Already sort of done, but needs to be in same plot, not separate
        subplots
    4. Code for barebones 3D plot in test.m. Same as 2D plot, the separate
        subplots need to be combined into a single colored plot
    5. Eigenvectors in U, L in L
%}

%Plot projections in 2d
col = {'rgbkcmy','o+x.+sdv^<>ph'};	% colors and markers for points

K = length(unique(Y));

figure(1); clf;
for j = 1:K
    Z = W'*X(Y==j, :)';
    scatter(Z(1,:), Z(2,:))
    hold on
    %axis([-3,8,-8,3])
end
hold off

predicted = knn(X, Y, hapt.test.feature_data, 1);
correct = predicted - hapt.test.activity_labels;
score = (correct == 0);
disp(sum(score)/3162);
