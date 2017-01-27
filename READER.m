function Fea_Order = READER( Xw, X, Y, opts )
%READER READER: Robust Semi-Supervised Multi-Label Dimension Reduction [1]
%
%    Syntax
%
%       Yt = READER( Xw,Y,Xt,opts )
%
%    Description
%
%       Input:
%           Xw           An N x M data matrix, each row denotes a sample
%           X            An n x M labeled data matrix, each row denotes a sample (n < N)
%           Y            An L x n label matrix, each column is a label set
%           Xt           An Nt x M test data matrix, each row is a test sample
%           opts         Parameters for READER
%             opts.alpha The factor on the feature coefficient matrix
%             opts.beta  The factor on manifold learning of Xw
%             opts.gamma The factor on manifold learning of Y
%             opts.k     The dimensionality of embedded label space
%             opts.p     The number of nearest neighbors
%             opts.b     The bias in loss function
%             opts.maxIt The maximum number of iterations
% 
%       Output
%           Yt           An L x Nt predicted label matrix, each column is a predicted label set
%
%  [1] L. Sun, M. Kudo and K. Kimura, READER: Robust Semi-Supervised Multi-Label Dimension Reduction. 
%  A submission to IEICE Trans. on Information and Systems.

%% Get the parameters
alpha = opts.alpha;
beta  = opts.beta;
gamma = opts.gamma;
k     = opts.k;
p     = opts.p;
b     = opts.b;
t_MAX = opts.maxIt;

%% Compute the Laplacian matrix
% Build the weighted graph for X 
opt_x.NeighborMode = 'KNN';
opt_x.k = p;
opt_x.WeightMode = 'HeatKernel';
Sx = constructW(Xw,opt_x);
Lx = diag(sum(Sx)) - Sx;
% Build the weighted graph for Y if necessary
if k < 1
    opt_y.NeighborMode = 'KNN';
    opt_y.k = p;
    opt_y.WeightMode = 'Cosine';
    Sy = constructW(Y',opt_y);
    Ly = diag(sum(Sy)) - Sy;
end

%% Absorb the bias b into X
if b == 1
    Xw = [Xw, ones(size(Xw,1),1)];
    X  = [X, ones(size(X,1),1)];
end

%% The alterating algorithm
L        = size(Y,1);
[n,M]    = size(X);
XLX      = beta*Xw'*Lx*Xw;
alpha2   = alpha.^2;
diag_Hn  = ones(n,1);
diag_Hm  = ones(M,1);
Q_val    = zeros(t_MAX,1);
t        = 1;
if opts.k == 1
    W = rand(M,L);
    while t <= t_MAX
        % Get Hn and Hm
        Hm  = diag(alpha2*diag_Hm);
        HnX = bsxfun(@times,X,diag_Hn);
        
        % Update W
        W   = (HnX'*X + Hm + XLX) \ (HnX' * Y');
        
        % Update H
        temp_Hn = sqrt(sum((X*W-Y').^2,2) + eps);
        temp_Hm = sqrt(sum(W.^2,2) + eps);
        
        % Until convergence
        Q_val(t) = sum(temp_Hn) + alpha*sum(temp_Hm) + beta*trace(W'*XLX*W);
        if (t>2 && (abs(Q_val(t)-Q_val(t-1))<1e-5 || abs(Q_val(t)-Q_val(t-1))/abs(Q_val(t-1))<1e-5))
            break;
        end
        diag_Hn = 0.5 ./ temp_Hn;
        diag_Hm = 0.5 ./ (alpha*temp_Hm);
        t = t + 1;
    end
else
    k     = round(k * L);
    W     = rand(M,k);  
    Ly_it = gamma * Ly;
    while t <= t_MAX
        % Get Hn and Hm
        Hm  = diag(alpha2*diag_Hm);
        Hn  = diag(diag_Hn);
        HnX = bsxfun(@times,X,diag_Hn);
        
        % Updata V
        V   = sparse(Hn + Ly_it) \ (HnX * W);
        
        % Update W
        W   = (HnX'*X + Hm + XLX) \ (HnX' * V);
        
        % Update H
        temp_Hn = sqrt(sum((X*W-V).^2,2) + eps);
        temp_Hm = sqrt(sum(W.^2,2) + eps);
        
        % Until convergence
        Q_val(t) = sum(temp_Hn) + alpha*sum(temp_Hm) + beta*trace(W'*XLX*W) + trace(V'*Ly_it*V);
        if (t>2 && (abs(Q_val(t)-Q_val(t-1))<1e-5 || abs(Q_val(t)-Q_val(t-1))/abs(Q_val(t-1))<1e-5))
            break;
        end
        diag_Hn = 0.5 ./ temp_Hn;
        diag_Hm = 0.5 ./ (alpha*temp_Hm);
        t = t + 1;
    end
end

%% Remove the last row of W (corresponding to b)
if b == 1
    W = W(1:(end-1),:);
end

%% Output the scores of features in the descending order
[~, Fea_Order] = sort(sum(W.^2,2),'descend');

end
