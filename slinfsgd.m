% [G,E,C] = slinfsgd(X,Y,Xv,Yv,o,eta,maxit,B,g0)
% Train logistic regression for binary classification y = s(w'.x+w0)
% with stochastic gradient descent, by maximum likelihood (min. cross-entropy)
% or by least-squares errors.
%
% In:
%   X: NxL matrix, N L-dim data points rowwise, training set inputs.
%   Y: Nx1 matrix, N labels in {0,1}, training set outputs.
%   Xv: MxL matrix, M L-dim data points rowwise, validation set inputs.
%   Yv: Mx1 matrix, M labels in {0,1}, validation set outputs.
%   o: objective function (0: maximum likelihood, 1: least-squares errors).
%   eta: positive scalar, the learning rate (step size).
%   maxit: number of iterations to run.
%   B: minibatch size, between 1 and N.
%   g0: (struct) initial logistic regressor (see definition in slinf.m).
% Out:
%   G: (ceil(N/B))x(maxit+1) cell array where G{n,i} is a struct containing
%      the logistic regressor at minibatch n and iteration i-1 (see struct
%      definition in slinf.m).
%   E: 2x(ceil(N/B))x(maxit+1) array where E(1,n,i) and E(2,n,i) contain the
%      training and validation error at minibatch n and iteration i-1.
%   C: 2x(ceil(N/B))x(maxit+1) array where C(1,n,i) and C(2,n,i) contain the
%      training and validation classification error (in [0,1]) at minibatch
%      n and iteration i-1.

function [G,E,C] = slinfsgd(X,Y,Xv,Yv,o,eta,maxit,B,g0)
if o==0
[N,~] = size(X); M= size(Xv,1); X= [X ones(N,1)];Xv=[Xv ones(M,1)];
w= (g0.w); w1= (g0.W); W = [w1 w]';sgdMaxitr= ceil(N/B);
E=zeros(2,ceil(N/B),maxit+1);
C=zeros(2,ceil(N/B),maxit+1);
G=cell(ceil(N/B),maxit+1);
for ii=1:sgdMaxitr
    G{ii,1}= g0;
    E(1,ii,1)= Error_sig(Y,W,X);
    E(2,ii,1) = Error_sig(Yv,W,Xv);
    C(1,ii,1)=sum(Y~=ceil(sig(W,X)-.5))/N;
    C(2,ii,1)=sum(Yv~=ceil(sig(W,Xv)-.5))/M;
end
for ii=1:maxit
    a=randperm(N);
    jj=1; J= X(a,:);Yj=Y(a,:);
    while(jj<=sgdMaxitr)
        
        if J(jj)*B>=N
            XB=J(((jj-1)*B+1):end,:);YB=Yj(((jj-1)*B+1):end,:);
            deltaW = eta*XB'*(YB-sig(W,XB));
        else
            XB=J(((jj-1)*B+1):jj*B,:);YB=Yj(((jj-1)*B+1):jj*B,:);
            deltaW = eta*XB'*(YB-sig(W,XB));
        end
        W= W+deltaW;
        g.w = W(end,:)'; g.W= W(1:end-1,:)';g.type= 'slinf';
        G{jj,ii+1}= g;
        E(1,jj,ii+1)= Error_sig(Y,W,X);
        E(2,jj,ii+1) = Error_sig(Yv,W,Xv); 
        C(1,jj,ii+1)=sum(Y~=ceil(sig(W,X)-.5))/N;
        C(2,jj,ii+1)=sum(Yv~=ceil(sig(W,Xv)-.5))/M;
        jj=jj+1;
    end
   % eta=eta/10;
end
end

if o==1
[N,~] = size(X); M= size(Xv,1); X= [X ones(N,1)];Xv=[Xv ones(M,1)];
w= (g0.w); w1= (g0.W); W = [w1 w]';sgdMaxitr= ceil(N/B);
E=zeros(2,ceil(N/B),maxit+1);
C=zeros(2,ceil(N/B),maxit+1);
G=cell(ceil(N/B),maxit+1);
for ii=1:sgdMaxitr
    G{ii,1}= g0;
    E(1,ii,1)= Error_lsqr(Y,W,X);
    E(2,ii,1) = Error_lsqr(Y,W,X);
    C(1,ii,1)=sum(Y~=ceil(sig(W,X)-.5))/N;
    C(2,ii,1)=sum(Yv~=ceil(sig(W,Xv)-.5))/M;
end
for ii=1:maxit
    a=randperm(N);
    jj=1; J= X(a,:);Yj=Y(a,:);
    while(jj<=sgdMaxitr)
        if J(jj)*B>=N
            XB=J(((jj-1)*B+1):end,:);YB=Yj(((jj-1)*B+1):end,:);
            deltaW = eta*XB'*((YB-sig(W,XB)).*sig(W,XB).*(1-sig(W.XB)));
        else
            XB=J(((jj-1)*B+1):jj*B,:);YB=Yj(((jj-1)*B+1):jj*B,:);
            deltaW = eta*XB'*((YB-sig(W,XB)).*sig(W,XB).*(1-sig(W,XB)));
        end
        W= W+deltaW;
        g.w = W(end,:)'; g.W= W(1:end-1,:)';g.type= 'slinf';
        G{jj,ii+1}= g;
        E(1,jj,ii+1)= Error_lsqr(Y,W,X);
        E(2,jj,ii+1) = Error_lsqr(Yv,W,Xv); 
        C(1,jj,ii+1)=sum(Y~=ceil(sig(W,X)-.5))/N;
        C(2,jj,ii+1)=sum(Yv~=ceil(sig(W,Xv)-.5))/M;
        jj=jj+1;
    end
   % eta=eta/10;
end
end
end


function [theta]=sig(W,X)

theta=(1+exp(-X*W)).^(-1);

end

function [E]=Error_sig(Y,W,X)
E=sum(-Y.*log(sig(W,X))-(1-Y).*log(1-sig(W,X)));
end

function [E]=Error_lsqr(Y,W,X)
E=sum(0.5*((Y-sig(W,X)).^2));
end