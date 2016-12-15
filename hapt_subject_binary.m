function [training, validation, test] = hapt_subject_binary(hapt, subject)
    [training, validation, test] = hapt_subject(hapt);
    
    Y = training.Y;
    Y(Y == subject) = 1;
    Y(Y ~= 1) = 0;
    training.Y = Y;
    
    Y = validation.Y;
    Y(Y == subject) = 1;
    Y(Y ~= 1) = 0;
    validation.Y = Y;
    
    Y = test.Y;
    Y(Y == subject) = 1;
    Y(Y ~= 1) = 0;
    test.Y = Y;
end
