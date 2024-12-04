function logRatio = CNNmodel_SimpleOptimized_FullData(channel1_index, channel2_index, data)
    % Simple NN implementation to compute and minimize univariate error using bivariate model.
    %
    % Parameters:
    %   channel1_index : Index of the first channel (target channel)
    %   channel2_index : Index of the second channel 
    %   data           : EEG data matrix 
    %
    % Returns:
    %   logRatio : Logarithmic ratio of error variances (univariate vs bivariate)
    
    % Extract channel data
    channel1Data = data(channel1_index, :);  
    channel2Data = data(channel2_index, :); 

    filterSize = 16;
    numOfFilters = 128;

    % Train Univariate Model    
    X_uni = channel1Data;  
    Y_uni = channel1Data;  

    % Reshape data for NN input
    XTrain_uni = reshape(X_uni, [], 1, 1);  
    YTrain_uni = reshape(Y_uni, [], 1);  
    XVal_uni = XTrain_uni;
    YVal_uni = YTrain_uni;

    % Define univariate network
    layers_uni = [
        sequenceInputLayer(1)
        convolution1dLayer(filterSize, numOfFilters * 2, 'Padding', 'same');
        reluLayer
        fullyConnectedLayer(1)  
    ];

    % Training options
    options_uni = trainingOptions('adam', ...
        'MaxEpochs', 200, ...
        'MiniBatchSize', 64, ...
        'L2Regularization', 0.001, ...
        'InitialLearnRate', 0.0001, ...
        'ValidationData', {XVal_uni, YVal_uni}, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots','training-progress');

    % Train the univariate model
    model_uni = trainnet(XTrain_uni, YTrain_uni, layers_uni, "mse", options_uni);

    % save('univariate_model.mat', 'model_uni');
    save('univariate_model_electrodePair.mat', 'model_uni');

    % Predict and compute error
    YPred_uni = predict(model_uni, XTrain_uni);
    error_uni = YTrain_uni - YPred_uni;
    var_uni = var(error_uni);  % Variance of univariate error
    mse_uni = mean(error_uni.^2);  % MSE of univariate error

    % Train Bivariate Model with Minimization
    % Prepare bivariate input and target
    X_bi = [channel1Data; channel2Data];  % Inputs: channel1 and channel2
    Y_bi = channel1Data;                  % Target: channel1

    % Reshape data for NN input
    XTrain_bi = reshape(X_bi', [], 2, 1);  
    YTrain_bi = reshape(Y_bi, [], 1);  
    XVal_bi = XTrain_bi;
    YVal_bi = YTrain_bi;

    % Define bivariate network
    layers_bi = [
        sequenceInputLayer(2)                             
        convolution1dLayer(filterSize, numOfFilters * 2, 'Padding', 'same');
        reluLayer
        fullyConnectedLayer(1)                             
    ];

    options_bi = trainingOptions('adam', ...
        'MaxEpochs', 400, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.0001, ...
         'L2Regularization', 0.001, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false,...
        'ValidationData', {XVal_bi, YVal_bi},...
        'Plots','training-progress');

    % Train the bivariate model
    model_bi = trainnet(XTrain_bi, YTrain_bi, layers_bi, "mse", options_bi);

    % save('bivariate_model.mat', 'model_bi');
    save('bivariate_model_electrodePair.mat', 'model_bi');

    % Predict and compute error
    YPred_bi = predict(model_bi, XTrain_bi);
    error_bi = YTrain_bi - YPred_bi;
    var_bi = var(error_bi);  % Variance of bivariate error
    mse_bi = mean(error_bi.^2);

    % Log Ratio 
    logRatio = log(var_uni / var_bi);

    % Display the variance results
    fprintf('Variance of Univariate Error: %.4f\n', var_uni);
    fprintf('Variance of Bivariate Error: %.4f\n', var_bi);
    fprintf('Log Ratio: %.4f\n', logRatio);

    % Check improvement
    if var_bi < var_uni
        disp("Bivariate model successfully minimized the error.");
    else
        disp("No improvement with the bivariate model.");
    end
end

