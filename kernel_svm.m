function [alphas,labels,landmarks] = kernel_svm(X, Y)
    [N,~] = size(X);
    
    H = (X*X').*(Y*Y');
    f = -ones(N,1);
    A = -eye(N);
    aTemp = zeros(N,1);
    B = [Y'; zeros(N-1,N)];
    b = zeros(N,1);

    [~,~,~,~,lambda] = quadprog(H+eye(N)*0.001,f,A,aTemp,B,b);
    alphas = lambda.ineqlin(find(lambda.ineqlin));
    labels = Y(find(lambda.ineqlin));
    landmarks = X(find(lambda.ineqlin));
end