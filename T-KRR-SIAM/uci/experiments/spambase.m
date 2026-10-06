%% Spambase Dataset
M = 30;
R = 10;
maxIte = 10;
lambda = 1e-5;
NTrials = 10;
trainErrorCP = zeros(NTrials,1);
testErrorCP = zeros(NTrials,1);
trainErrorKRR = zeros(NTrials,1);
testErrorKRR = zeros(NTrials,1);
nll = zeros(NTrials, 1);
warning('off','all');
for ite = 1:NTrials
    rng(ite);
    X = readmatrix('spambase.csv');
    perm = randperm(size(X,1));
    X = X(perm,:);
    X = X(1:floor(0.9*size(X,1)),:);
    Y = X(:,end);
    X = X(:,1:end-1);
    Y = (Y==1)-(Y==0); 
    XMean = mean(X);  XStd = std(X);
    X = (X-XMean)./XStd;
    lengthscale = mean(std(X));
    
    %CP
    WCP = CPLS(X,Y,M,R,lambda,maxIte);
    trainErrorCP(ite) = mean(Y~=sign(CPPredict(X,WCP)));
    
    % KRR
    [wKRR,XTrain] = KRR(X,Y,lengthscale,lambda);
    trainErrorKRR(ite) = mean(Y~=sign(SE(X,X,lengthscale)*wKRR));

    % Test
    X = readmatrix('spambase.csv');
    X = X(perm,:);
    X = X(floor(0.9*size(X,1))+1:end,:);
    Y = X(:,end);
    X = X(:,1:end-1);
    Y = (Y==1)-(Y==0); 
    X = (X-XMean)./XStd;
    testErrorCP(ite) = mean(Y~=sign(CPPredict(X,WCP)));
    testErrorKRR(ite) = mean(Y~=sign(SE(X,XTrain,lengthscale)*wKRR));

    %nll 
    K_star = SE(X, XTrain, lengthscale);               % K(X_test, X_train)
    pred_mean = K_star * wKRR;                         % f* = K_* w
    
    % Predictive variance
    K_train = SE(XTrain, XTrain, lengthscale);         % K(X_train, X_train)
    L = chol(K_train + lambda * eye(size(K_train)), 'lower');
    v = L \ K_star';                                   % Solve L v = K_star'
    sigma2 = diag(SE(X, X, lengthscale)) - sum(v.^2, 1)';  % diag(K**) - vᵀv
    pred_std = sqrt(max(sigma2, 1e-10));               % Numerical safety

    Y_binary = (Y + 1) / 2;
    probs_gt_zero = 1 - normcdf(0, pred_mean, pred_std);%P(y > 0)
    eps = 1e-15;
    probs_clipped = max(min(probs_gt_zero, 1 - eps), eps);

    nll(ite) = -mean(Y_binary .* log(probs_clipped) + ...
                        (1 - Y_binary) .* log(1 - probs_clipped));
end


