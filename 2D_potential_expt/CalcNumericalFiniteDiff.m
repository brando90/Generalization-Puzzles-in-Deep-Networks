function [ vNumericalGrad ] = CalcNumericalFiniteDiff( vInputPoint, hInputFunc, epsVal )

numElmnts       = size(vInputPoint, 2);
vNumericalGrad  = zeros([1,numElmnts]);

refVal = hInputFunc(vInputPoint);

for ii = 1:numElmnts
    % Set the perturbation vector
    refInVal = vInputPoint(ii);
    vInputPoint(ii) = refInVal + epsVal;

    % Compute Numerical Gradient
    vNumericalGrad(ii) = (hInputFunc(vInputPoint) - refVal) / epsVal;
    %vNumericalGrad(ii) = (hInputFunc(vInputPoint) - refVal);

    % Reset the perturbation vector
    vInputPoint(ii) = refInVal;
end

end