%% Adult Dataset
M = 30;
R = 10;
maxIte = 10;
lambda = 1e-5;
NTrials = 10;
trainErrorCP = zeros(NTrials,1);
testErrorCP = zeros(NTrials,1);
timeCP = zeros(NTrials,1);
warning('off','all');
for ite = 1:NTrials
    rng(ite);
    X = readmatrix('adult.csv');
    perm = randperm(size(X,1));
    X = X(perm,:);
    X = X(1:floor(0.9*size(X,1)),:);
    Y = X(:,end);
    X = X(:,1:end-1);

    Y = (Y==1)-(Y==0); 
    XMean = mean(X);  XStd = std(X);
    
    X = (X-XMean)./XStd;
    lengthscale = mean(std(X));
    tic;

    WCP = CPLS(X,Y,M,R,lambda,maxIte);
    timeCP(ite) = toc;
    trainErrorCP(ite) = mean(Y~=sign(CPPredict(X,WCP)));
    
    
    % Test
    X = readmatrix('adult.csv');
    X = X(perm,:);
    X = X(floor(0.9*size(X,1))+1:end,:);
    Y = X(:,end);
    X = X(:,1:end-1);
    Y = (Y==1)-(Y==0); 
    X = (X-XMean)./XStd;
    testErrorCP(ite) = mean(Y~=sign(CPPredict(X,WCP)));
end

