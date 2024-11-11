function logRatio = ARmodel_FullData(channel1, channel2, data)
    % Computes the log ratio of error variances between two channels.
    %
    % Parameters:
    %   channel1 : int, index of the first channel
    %   channel2 : int, index of the second channel
    %   data     : matrix, EEG data with channels as rows
    %
    % Returns:
    %   logRatio : float, logarithmic error variance ratio

    order = 16; % Model order

    % Prepare data for univariate and bivariate models
    inputData1 = data(channel1, :)';
    inputData2 = data(channel2, :)';

    Y_target = inputData1(order+1:end); % Target values

    % Create lagged predictors for univariate model on channel1
    X_uni = create_lagged_matrix(inputData1, order);

    % Fit univariate model and compute prediction error
    error_uni = compute_model_error(X_uni, Y_target);

    % Create lagged predictors for bivariate model on channels1 and2
    X_bi = create_lagged_matrix([inputData1, inputData2], order);

    % Fit bivariate model and compute prediction error
    error_bi = compute_model_error(X_bi, Y_target);

    % Log ratio of variances
    logRatio = log(var(error_uni) / var(error_bi));

    % Calculate and display impulse response (optional, for this pair)
    %impulseResponse = calculate_impulse_response(coefficients, order);
    % disp('Impulse Response for the bivariate model:');
    % disp(impulseResponse);

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

function error = compute_model_error(X, Y)
    % Fits an autoregressive model and returns the prediction error.
    %
    % Parameters:
    %   X : matrix, predictors with lagged values
    %   Y : vector, target values
    %
    % Returns:
    %   error : vector, prediction error

    coefficients = (X' * X) \ (X' * Y); % Model coefficients
    Y_pred = X * coefficients;          % Predicted values
    error = Y - Y_pred;                 % Prediction error
end

function impulseResponse = calculate_impulse_response(coefficients, order)
    % Calculates the impulse response for an autoregressive model.
    %
    % Parameters:
    %   coefficients : vector, AR model coefficients for bivariate model
    %   order        : int, model order (number of lags)
    %
    % Returns:
    %   impulseResponse : vector, impulse response over time
    
    % Reshape coefficients to separate for each lag and channel
    numChannels = 2; % Assuming bivariate model (2 channels)
    coefficients = reshape(coefficients, order, numChannels);

    % Initialize impulse response (identity for the first time step)
    impulseResponse = eye(numChannels);

    % Calculate impulse response iteratively
    for t = 2:order
        impulseResponse(:, t) = coefficients(t, :)' * impulseResponse(:, t - 1);
    end
end



