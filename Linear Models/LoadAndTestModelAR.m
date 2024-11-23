function logRatio = LoadAndTestModelAR(channel1_index, channel2_index, testData)
    % LoadAndTestModelAR: Compute the log ratio of error variances using trained AR models.
    %
    % Parameters:
    %   channel1_index : Index of the first channel (target channel)
    %   channel2_index : Index of the second channel
    %   testData       : Test data matrix [channels x samples]
    %
    % Returns:
    %   logRatio       : Logarithmic ratio of error variances (univariate vs bivariate)
    
    % Load AR model coefficients
    load('AR_Univariate_model.mat', 'coefficients_uni');
    load('AR_Bivariate_model.mat', 'coefficients_bi');
    
    order = 16; % Model order used during training
    
    % Extract data for specified channels
    channel1Data = testData(channel1_index, :)';
    channel2Data = testData(channel2_index, :)';

    % Prepare target data
    Y_target = channel1Data(order+1:end);

    % Create lagged predictors for the univariate model
    X_uni = create_lagged_matrix(channel1Data, order);

    % Compute predictions for univariate model
    YPred_uni = X_uni * coefficients_uni;

    % Compute error for univariate model
    error_uni = Y_target - YPred_uni;
    var_uni = var(error_uni);

    % Create lagged predictors for the bivariate model
    X_bi = create_lagged_matrix([channel1Data, channel2Data], order);

    % Compute predictions for bivariate model
    YPred_bi = X_bi * coefficients_bi;

    % Compute error for bivariate model
    error_bi = Y_target - YPred_bi;
    var_bi = var(error_bi);

    % Compute and display the log ratio of variances
    logRatio = log(var_uni / var_bi);
    disp(['Variance of Univariate Error: ', num2str(var_uni)]);
    disp(['Variance of Bivariate Error: ', num2str(var_bi)]);
    disp(['Log Ratio: ', num2str(logRatio)]);
end

function laggedMatrix = create_lagged_matrix(data, order)
    % Creates a lagged matrix for autoregressive model input.
    %
    % Parameters:
    %   data  : matrix, column vectors of input data for each channel
    %   order : int, model order (number of lags)
    %
    % Returns:
    %   laggedMatrix : matrix, matrix with lagged values for autoregressive model

    n = size(data, 1);
    numVars = size(data, 2);
    laggedMatrix = zeros(n - order, order * numVars);

    for i = 1:order
        laggedMatrix(:, (i-1)*numVars + (1:numVars)) = data(order+1-i:end-i, :);
    end
end
