%% Naval Dataset
M = 20;
R = 10;
maxIte = 10;
NTrials = 10;
trainErrorCP = zeros(NTrials,1);
testErrorCP = zeros(NTrials,1);
trainErrorGP = zeros(NTrials,1);
testErrorGP = zeros(NTrials,1);
nll_gp = zeros(NTrials,1);
warning('off','all');
for ite = 1:NTrials
    rng(ite);
    X = readmatrix('naval.csv');
    perm = randperm(size(X,1));
    X = X(perm,:);
    X = X(1:floor(0.90*size(X,1)),:);
    Y = X(:,end);
    X = X(:,1:end-1);

    YMean = mean(Y);    YStd = std(Y);
    XMean = mean(X);  XStd = std(X);
    Y = (Y-YMean)./YStd;
    X = (X-XMean)./XStd;
    meanfunc = [];                    
    covfunc = @covSEiso;         
    likfunc = @likGauss;

    % GP
    hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
    hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc,covfunc,likfunc,X,Y);
    trainErrorGP(ite) = mean((gp(hyp2,@infGaussLik,meanfunc,covfunc,likfunc,X,Y,X)-Y).^2);
    lengthscale = exp(hyp2.cov(1));
    lambda = exp(hyp2.lik-hyp2.cov(2))^2;

    
    WCP = CPLS(X,Y,M,R,lambda,maxIte);
    trainErrorCP(ite) = mean((Y-CPPredict(X,WCP)).^2);

    % Test
    XTest = readmatrix('naval.csv');
    XTest = XTest(perm,:);
    XTest = XTest(floor(0.90*size(XTest,1))+1:end,:);
    YTest = XTest(:,end);
    XTest = XTest(:,1:end-1);
    XTest = (XTest-XMean)./XStd;

    YTestPred_CP_std = CPPredict(XTest, WCP);
    [YTestPred_GP_std, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, X, Y, XTest);
    sigma = sqrt(s2) * YStd;  % Unstandardize std
    
    % Unstandardize predictions and ground truth
    YTestPred_CP = YTestPred_CP_std * YStd + YMean;
    YTestPred_GP = YTestPred_GP_std * YStd + YMean;
    YTest_actual = YTest; 
    
    % RMSE
    testErrorCP(ite) = sqrt(mean((YTest_actual - YTestPred_CP).^2));
    testErrorGP(ite) = sqrt(mean((YTest_actual - YTestPred_GP).^2));
    nll_gp(ite) = mean(0.5*log(2*pi*sigma.^2) + 0.5*((YTest_actual - YTestPred_GP).^2)./sigma.^2);

end


