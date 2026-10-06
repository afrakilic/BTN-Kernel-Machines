function Mati = p_features(X, input_dimension)
% Pure-Power Polynomial Features
%
% Parameters:
% X : vector (N x 1)
%     Input data (column vector)
% input_dimension : int
%     Maximum power (degree) of the polynomial features
%
% Returns:
% Mati : matrix (N x input_dimension)
%     Unit-norm pure-power features

% Compute the pure-power features
Mati = X .^ (0:input_dimension-1); % Generate powers of X

% Normalize each row to have unit norm
norms = vecnorm(Mati, 2, 2); % Compute L2 norm along rows
Mati = (Mati ./ norms) + 0.2; % Normalize each row

end