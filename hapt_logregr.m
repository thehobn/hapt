function [tpr, fpr] = hapt_logregr(training, test, o, eta, maxit, B, g0)
    D = size(training.X, 2);
    [~,~,C] = slinfsgd(training.X, training.Y, test.X, test.Y, ...
        o, eta, maxit, B, g0);
    % pretend like this classifier always picks the opposite
    fpr = mean(C(2, :, maxit));
    tpr = 1 - fpr;
end
