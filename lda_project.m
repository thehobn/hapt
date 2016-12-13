function [W, lswcSM, lsbcSM, Z] = lda_project(Ev, J, L, wcSM, bcSM, X)
    % Inputs:
    %   DxD Ev with right eigenvectors as columns
    %   Dx2 J eigenvectors sorted by Fisher discriminant
    %       J(:,1) indexes
    %       J(:,2) Fisher discriminants
    %   1x1 L number of dimensions to have in subspace
    %   DxD wcSM within-class scatter matrix
    %   DxD bcSM between-class scatter matrix
    %   DxN X data
    % Outputs:
    %   DxL W containing L best eigenvectors from Ev
    %   LxL lsbcSM latent space between-class scatter matrix
    %   LxL lswcSM latent space within-class scatter matrix
    %   LxN Z with points in X projected onto LDA subspace of dimension L

    % Take L best eigenvectors
    W = Ev( :, J( 1:L, 1 ) );
    
    % Compute latent space scatter matrices
    lswcSM = W' * wcSM * W;
    lsbcSM = W' * bcSM * W;
    
    % Project data onto subspace
    Z = W'*X;
end
