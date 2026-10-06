function score = CPPredict(X, W)
    [N,D] = size(X);
    M = size(W{1},1);
    score = ones(N,1);
    for d = 1:D
        score = score.*(p_features(X(:,d),M)*W{d});
    end
    score = sum(score,2);
end