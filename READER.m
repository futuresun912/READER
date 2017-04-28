function idF = READER( Xw, X, Y, opts )
%READER Robust Semi-Supervised Multi-Label Dimension Reduction [1]
% The efficient optimization algorithm is used in this program.
%
%    Syntax
%
%       idF = READER( Xw,X,Y,opts )
%
%    Description
%
%       Input:
%           Xw           An N x M data matrix, each row denotes a sample
%           X            An n x M data matrix, each row denotes a sample (n < N)
%           Y            An L x n label matrix, each column is a label set
%           opts         Parameters of READER
%             opts.alpha The factor on the projection matrix
%             opts.beta  The factor on manifold learning of Xw
%             opts.gamma The factor on manifold learning of Y
%             opts.k     The dimensionality of embedded label space
%             opts.p     The number of nearest neighbors
%             opts.maxIt The maximum number of iterations
%             opts.epsIt The minimum value of relative change
%
%       Output
%           idF          The indices of features in descending order
%
%  [1] READER: Robust Semi-Supervised Multi-Label Dimension Reduction.

%% Get the parameters
alpha = opts.alpha;
beta  = opts.beta;
gamma = opts.gamma;
k     = opts.k;
p     = opts.p;
maxIt = opts.maxIt;
epsIt = opts.epsIt;

%% Compute the Laplacian matrix
tic;
[numNn,numF] = size(Xw);
[numL,numN] = size(Y);
% Build the weighted graph for X
if beta > 0
    opt_x.NeighborMode = 'KNN';
    opt_x.k = p;
    opt_x.t = 1;
    opt_x.WeightMode = 'HeatKernel';
    Sx = constructW(Xw,opt_x);
    Dx = sparse(1:numNn,1:numNn,sum(Sx),numNn,numNn);
    Lx = Dx - Sx;
else 
    Lx = 0;
end
% Build the weighted graph for Y
if gamma > 0
    opt_y.NeighborMode = 'KNN';
    opt_y.k = p;
    opt_y.WeightMode = 'Cosine';
    Sy = constructW(Y',opt_y);
    Dy = sparse(1:numN,1:numN,sum(Sy),numN,numN);
    Ly = Dy - Sy;
end

%% Initialization
XLX = beta*Xw'*Lx*Xw;
XLX = max(XLX,XLX');
Q_val = zeros(maxIt,1);
alphaIt = alpha.^2;
t = 1;
if gamma > 0
    % Initialization
    gLy = gamma.*Ly;
    k   = round(k*numL);
    diagH = ones(numN+numF,1);
    W = rand(numF,k);
    while t < maxIt
        % Get Hn and Hm
        diagHn = diagH(1:numN);
        diagHm = diagH(numN+1:end);
        Hm  = sparse(1:numF,1:numF,alphaIt.*diagHm,numF,numF);
        Hn  = sparse(1:numN,1:numN,diagHn,numN,numN);
        HnX = bsxfun(@times,X,diagHn);
        
        % Updata V and W
        V = (Hn+gLy) \ (HnX*W);
        W = (HnX'*X+Hm+XLX) \ (HnX'*V);
        
        % Update H
        AU = [X*W-V; alpha*W];
        tmpAU = sum(AU.^2,2);
        diagH = 0.5 ./ sqrt(tmpAU + eps);
        
        % Check if convergence condition is matched
        Q_val(t) = sum(sqrt(tmpAU)) + trace(W'*XLX*W) + trace(V'*gLy*V);
        if t > 1
            if ( abs(Q_val(t)-Q_val(t-1))<epsIt || abs(Q_val(t)-Q_val(t-1))/Q_val(t-1)<epsIt )
                t = t + 1;
                break;
            end
        end
        t = t + 1;
    end
else
    % Initialization
    Y = Y';
    diagHn  = ones(numN,1);
    diagHm  = ones(numF,1);
    while t < maxIt
        % Get Hn and Hm
        Hm  = diag(alphaIt*diagHm);
        HnX = bsxfun(@times,X,diagHn);
        
        % Update W
        W = (HnX'*X+Hm+XLX) \ (HnX'*Y);
        
        % Update H
        tmpXWY = sum((X*W-Y).^2,2);
        tmpW   = sum((alpha*W).^2,2);
        diagHn = 0.5 ./ sqrt(tmpXWY + eps);
        diagHm = 0.5 ./ sqrt(tmpW + eps);
        
        %% Convergence condition
        Q_val(t) = sum(sqrt(tmpXWY)) + sum(sqrt(tmpW)) + trace(W'*XLX*W);
        if t > 1
            if ( abs(Q_val(t)-Q_val(t-1))<epsIt || abs(Q_val(t)-Q_val(t-1))/Q_val(t-1)<epsIt )
                t = t + 1;
                break;
            end
        end
        t = t + 1;
    end
end
disp(['READER converged at the ',num2str(t),'th iteration with ',...
    num2str(Q_val(t-1)),'. Time cost: ',num2str(toc),'s']);

%% Return the indices of ranked features
[~,idF] = sort(sum(W.^2,2),'descend');

end