function [tpr, fpr] = hapt_linsvm(train, test, C, dual)
    [f,~,~,~,~,~] = linsvmtrain(train.X, train.Y, C, dual);
    prediction = linsvm(test.X, f);

    truth = test.Y;
    tpr = sum(prediction==truth)/numel(truth);
    fpr = 1 - tpr;
end
sort
