function L = kernel_disc(alphas, labels, landmarks, x, sigmaSquared)
    g = 0;
    for i = 1:size(alphas,1)
        g = g + (alphas(i) * labels(i) * exp(-0.5 * norm(landmarks(i) - x)^2 / sigmaSquared));
    end
    
    if (g < 0)
        L = -1;
    else
        L = 1;
    end
end