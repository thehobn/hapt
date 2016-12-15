function [training, test] = hapt_activity_binary_lda(hapt, L)
    [training, test] = hapt_activity(hapt);
    
    X = training.X;
    Y = training.Y;
    Y(Y <= 3) = 0; % walking
    Y(Y >= 7) = 1;  % transition
    X = X((Y == 0 | Y == 1),:);
    Y = Y((Y == 0 | Y == 1),:);
    training.X = X;
    training.Y = Y+1;
    
    X = test.X;
    Y = test.Y;
    Y(Y <= 3) = 0; % walking
    Y(Y >= 7) = 1;  % transition
    X = X((Y == 0 | Y == 1),:);
    Y = Y((Y == 0 | Y == 1),:);
    test.X = X;
    test.Y = Y+1;
    
    [~,Sw,~,Sb,U,~,J] = lda(training.X, training.Y);
    [W,~,~,training.X] = lda_project(U, J, L, Sb, Sw, training.X);
    test.X = test.X * W;
    
end
