function [training, test] = hapt_activity_lda(hapt, L)
    [training, test] = hapt_activity(hapt);

    [~,Sw,~,Sb,U,~,J] = lda(training.X, training.Y);
    [W,~,~,training.X] = lda_project(U, J, L, Sb, Sw, training.X);
    test.X = test.X * W;
end
