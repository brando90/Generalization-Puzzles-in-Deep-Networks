function [ Kern ] = get_poly_feat_1D( X,D )
% X = raw input features
[N,~] = size(X); % [D,1]
Kern = zeros(N,D+1);
for n=1:N
    x = X(n,1);
    for d=1:D+1
        deg=d-1;
        Kern(n,d) = x^deg;
    end
end
end

