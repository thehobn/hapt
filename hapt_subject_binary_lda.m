function [training, validation, test] = hapt_subject_binary_lda(hapt, L, subject)
    [training, validation, test] = hapt_subject_binary(hapt, subject);
    
    [~,Sw,~,Sb,U,~,J] = lda(training.X, training.Y);
    [W,~,~,training.X] = lda_project(U, J, L, Sb, Sw, training.X);
    validation.X = validation.X * W;
    test.X = test.X * W;
end
