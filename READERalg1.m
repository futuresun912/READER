function idF = READERalg1( Xw, X, Y, opts )
%READERalg1 Robust Semi-Supervised Multi-Label Dimension Reduction [1]
% The Algorithm 1 is used for READER in this program.
%
%    Syntax
%
%       idF = READERalg1( Xw,X,Y,opts )
%
%    Description
%
%       Input:
%           Xw           An N x M data matrix, each row denotes a sample
%           X            An n x M data matrix, each row denotes a sample (n < N)
%           Y            An L x n label matrix, each column is a label set
%           opts         Parameters of READERalg1
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
opt_x.NeighborMode = 'KNN';
opt_x.k = p;
opt_x.t = 1;
opt_x.WeightMode = 'HeatKernel';
Sx = constructW(Xw,opt_x);
Dx = sparse(1:numNn,1:numNn,sum(Sx),numNn,numNn);
Lx = Dx - Sx;
% Build the weighted graph for Y
opt_y.NeighborMode = 'KNN';
opt_y.k = p;
opt_y.WeightMode = 'Cosine';
Sy = constructW(Y',opt_y);
Dy = sparse(1:numN,1:numN,sum(Sy),numN,numN);
Ly = Dy - Sy;

%% Initialization
XLX = beta*Xw'*Lx*Xw;
XLX = max(XLX,XLX');
Q_val = zeros(maxIt,1);
gLy = gamma.*Ly;
zMat = sparse(zeros(numF,numN));
A = [X,-speye(numN); alpha.*speye(numF),zMat];
B = [XLX,zMat; zMat',gLy];
XDX = Xw'*bsxfun(@times,Xw,sum(Sx,2));
XDX = max(XDX,XDX');
E = [XDX,zMat; zMat',Dy];
k = round(k*numL);
diagH = ones(numN+numF,1);

%% The alterating algorithm
t = 1;
while t < maxIt
    % Update F
    F = A'*bsxfun(@times,A,diagH) + B;
    F = max(F,F');
    
    % Solve the eigenvalue problem
    [U,~] = eigs(E,F,k);
    W = U(1:numF,:);
    V = U(numF+1:end,:);
    
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
disp(['READERalg1 converged at the ',num2str(t),'th iteration with ',...
    num2str(Q_val(t-1)),'. Time: ',num2str(toc),'s']);

%% Select the top ranked features 
[~, idF] = sort(sum(W.^2,2),'descend');

end