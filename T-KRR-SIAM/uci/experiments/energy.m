%% Energy Dataset
M = 20;
R = 10;
maxIte = 10;
NTrials = 10;
trainErrorCP = zeros(NTrials,1);
testErrorCP = zeros(NTrials,1);
trainErrorGP = zeros(NTrials,1);
testErrorGP = zeros(NTrials,1);
nll_gp = zeros(NTrials,1);
all_y_test = [];
all_pred_mean = [];
all_pred_std = [];
all_run_ids = [];
warning('off','all');
X_full = readmatrix('enerji.csv');
Y_full = X_full(:, end);
X_full = X_full(:, 1:end-1);
for ite = 1:NTrials
    rng(ite);
    perm  = randperm(size(X_full, 1));
    split = floor(0.90 * size(X_full, 1));
    X     = X_full(perm(1:split),     :);
    Y     = Y_full(perm(1:split));
    XTest = X_full(perm(split+1:end), :);
    YTest = Y_full(perm(split+1:end));
% Normalize using train stats
    YMean = mean(Y);  YStd = std(Y);
    XMean = mean(X);  XStd = std(X);
    Y     = (Y - YMean) ./ YStd;
    X     = (X - XMean) ./ XStd;
    XTest = (XTest - XMean) ./ XStd;
    meanfunc = [];
    covfunc  = @covSEiso;
    likfunc  = @likGauss;
% GP
    hyp  = struct('mean', [], 'cov', [0 0], 'lik', -1);
    hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, X, Y);
    trainErrorGP(ite) = mean((gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, X, Y, X) - Y).^2);
    lengthscale = exp(hyp2.cov(1));
    lambda      = exp(hyp2.lik - hyp2.cov(2))^2;
    WCP         = CPLS(X, Y, M, R, lambda, maxIte);
    trainErrorCP(ite) = mean((Y - CPPredict(X, WCP)).^2);
% Predict
    YTestPred_CP_std       = CPPredict(XTest, WCP);
    [YTestPred_GP_std, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, X, Y, XTest);
    sigma                  = sqrt(s2) * YStd;
% Unstandardize
    YTestPred_CP = YTestPred_CP_std * YStd + YMean;
    YTestPred_GP = YTestPred_GP_std * YStd + YMean;
    YTest_actual = YTest;
    all_y_test    = [all_y_test;    YTest_actual];
    all_pred_mean = [all_pred_mean; YTestPred_GP];
    all_pred_std  = [all_pred_std;  sigma];
    all_run_ids   = [all_run_ids;   repmat(ite, length(YTest_actual), 1)];
% RMSE & NLL
    testErrorCP(ite) = sqrt(mean((YTest_actual - YTestPred_CP).^2));
    testErrorGP(ite) = sqrt(mean((YTest_actual - YTestPred_GP).^2));
    nll_gp(ite)      = mean(0.5*log(2*pi*sigma.^2) + 0.5*((YTest_actual - YTestPred_GP).^2)./sigma.^2);
end
T = table(all_run_ids, all_y_test, all_pred_mean, all_pred_std, ...
'VariableNames', {'run', 'y_test', 'prediction_mean', 'prediction_std'});
writetable(T, 'energy_all_runs_gp.csv');