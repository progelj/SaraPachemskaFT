% Initialize the sum matrix with zeros, assuming all matrices have the same size
numMatrices = numel(logRatioMatrices);
matrixSize = size(logRatioMatrices{1});
averageMatrix_t1_t2 = zeros(matrixSize);
% averageMatrix_t0 = zeros(matrixSize);

% Loop over all matrices and sum them up
for idx = 1:numMatrices
    currentMatrix = logRatioMatrices{idx};
    % Accumulate the matrices, ignoring NaN values
    averageMatrix_t1_t2 = averageMatrix_t1_t2 + nan_to_zero(currentMatrix);
end

% Divide by the number of matrices to get the average, ignoring NaN contributions
validCounts = zeros(matrixSize);
for idx = 1:numMatrices
    validCounts = validCounts + ~isnan(logRatioMatrices{idx});
end
averageMatrix_t1_t2 = averageMatrix_t1_t2 ./ validCounts;


function matrix = nan_to_zero(matrix)
    % Helper function to replace NaN values with zeros
    matrix(isnan(matrix)) = 0;
end
