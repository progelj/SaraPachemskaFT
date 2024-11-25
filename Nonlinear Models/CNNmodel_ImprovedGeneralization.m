function logRatio = CNNmodel_ImprovedGeneralization(channel1_index, channel2_index, dataTrain, dataTest)
    % CNN Model with Improved Generalization for EEG Data
    %
    % Parameters:
    %   channel1_index : Index of the first channel (target channel)
    %   channel2_index : Index of the second channel 
    %   dataTrain      : EEG training data matrix 
    %   dataTest       : EEG testing data matrix
    %
    % Returns:
    %   logRatio : Logarithmic ratio of error variances (univariate vs bivariate)
    
    % Extract channel data for training and testing
    channel1Train = dataTrain(channel1_index, :);  
    channel2Train = dataTrain(channel2_index, :); 
    channel1Test = dataTest(channel1_index, :);  
    channel2Test = dataTest(channel2_index, :); 

    % Parameters
    filterSize = 16;
    numOfFilters = 128;  

    % Train Univariate Model    
    XTrain_uni = reshape(channel1Train, [], 1, 1);  
    YTrain_uni = reshape(channel1Train, [], 1);  
    XVal_uni = reshape(channel1Test, [], 1, 1);  
    YVal_uni = reshape(channel1Test, [], 1);

    % Define univariate network
    layers_uni = [
        sequenceInputLayer(1)
        convolution1dLayer(filterSize, 64, 'Padding', 'same');
        reluLayer
        fullyConnectedLayer(1)  
    ];

    % Training options
    options_uni = trainingOptions('adam', ...
        'MaxEpochs', 400, ...
        'MiniBatchSize', 64, ...
        'L2Regularization', 0.001, ...
        'InitialLearnRate', 0.0001, ...
        'ValidationData', {XVal_uni, YVal_uni}, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    % Train the univariate model
    model_uni = trainnet(XTrain_uni, YTrain_uni, layers_uni, "mse", options_uni);
    
    save('univariate_model.mat', 'model_uni');

    % Predict and compute error on test set
    YPred_uni = predict(model_uni, XVal_uni);
    error_uni = YVal_uni - YPred_uni;
    var_uni = var(error_uni);  % Variance of univariate error

    % Train Bivariate Model
    % Prepare bivariate input and target
    XTrain_bi = [channel1Train; channel2Train]';  % Inputs: channel1 and channel2
    YTrain_bi = channel1Train;                   % Target: channel1
    XVal_bi = [channel1Test; channel2Test]';     % Validation inputs
    YVal_bi = channel1Test;                      % Validation target

    % Reshape data for NN input
    XTrain_bi = reshape(XTrain_bi, [], 2, 1);  
    YTrain_bi = reshape(YTrain_bi, [], 1);  
    XVal_bi = reshape(XVal_bi, [], 2, 1);  
    YVal_bi = reshape(YVal_bi, [], 1);

    % Define bivariate network
    layers_bi = [
        sequenceInputLayer(2)                             
        convolution1dLayer(filterSize, numOfFilters, 'Padding', 'same');
        reluLayer
        fullyConnectedLayer(1)                             
    ];

    options_bi = trainingOptions('adam', ...
        'MaxEpochs', 400, ...
        'MiniBatchSize', 64, ...
        'L2Regularization', 0.001, ...
        'InitialLearnRate', 0.0001, ...
        'ValidationData', {XVal_bi, YVal_bi}, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    % Train the bivariate model
    model_bi = trainnet(XTrain_bi, YTrain_bi, layers_bi, "mse", options_bi);

    save('bivariate_model.mat', 'model_bi');

    % Predict and compute error on test set
    YPred_bi = predict(model_bi, XVal_bi);
    error_bi = YVal_bi - YPred_bi;
    var_bi = var(error_bi);  % Variance of bivariate error

    % Compute Log Ratio and Display Results
    logRatio = log(var_uni / var_bi);
    fprintf('Variance of Univariate Error: %.4f\n', var_uni);
    fprintf('Variance of Bivariate Error: %.4f\n', var_bi);
    fprintf('Log Ratio: %.4f\n', logRatio);

    if var_bi < var_uni
        disp("Bivariate model successfully minimized the error.");
    else
        disp("No improvement with the bivariate model.");
    end
end
