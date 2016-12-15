function [tpr, fpr] = hapt_knn(train, test, K)
    prediction = knn(train.X, train.Y, test.X, K);

    truth = test.Y;
    tpr = sum(prediction==truth)/numel(truth);
    fpr = 1 - tpr;
end
