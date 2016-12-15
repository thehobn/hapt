function [wckSM, twcSM, Xmk, bcSM, Ev, Ed, J] = lda(X, Y)
    % Inputs:
    %   NxD X data
    %   1x1 D dimensions in data
    %   Nx1 Y labels
    %   1x1 K number of classes
    
    X = X';
    % im so sorry
    Y = Y';
    
    [D, ~] = size(X);
    C = unique(Y);
    K = length(C);

    % Outputs:
    %   DxDxK wckSM
    DD = sqrt (D);
    wckSM = zeros(D,D,K);
    %   DxD twcSM within-class scatter matrix
    twcSM = zeros(D,D);
    %   DxK Xmk means of each dimension in X for each class
    Xmk = zeros(D,K);
    %   DxD bcSM between-class scatter matrix
    %   DxD Ev with right eigenvectors as columns
    %   DxD Ed with eigenvalues on diagonal
    %   Dx2 J eigenvectors sorted by Fisher discriminant
    %       J(:,1) indexes
    %       J(:,2) Fisher discriminants
    J = zeros(D,2);
    
    %{
      Compute {within,between}-class scatter matrices
    %}
    for k = 1:K
      wckSM(:,:,k) = cov( X( :, Y==C(k) )' );
      twcSM = twcSM + wckSM(:,:,k);
      Xmk(:,k) = mean( X( :, Y==C(k) ), 2 );
      %{
      figure(k + K); clf; colormap(gray(256)); imagesc(reshape((Xmk(:,k)),DD,DD),[0 1]);
      set(gca,'XTick',[],'YTick',[]); axis image;
      set(gcf,'Name','mean of the data');
      %}
    end
    twcSM = twcSM + diag( repmat( 0.0000000001 * trace(twcSM) / D, 1, D ) );
    bcSM = cov(Xmk');

    % Eigendecompose
    [Ev, Ed] = eig( bcSM, twcSM );
    
    %{
      Sort eigenvectors by Fisher discriminant
    %}
    for d = 1:D
        W = Ev(:,d);
        J(d,2) = det( W' * bcSM * W ) / det( W' * twcSM * W );
    end
    [J(:,2), J(:,1)] = sort( J(:,2), 'descend' );
end
