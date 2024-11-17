function logRatio = LoadAndTestModel(channel1_index, channel2_index, testData)
    % computeLogRatio: Compute the log ratio of error variances using trained CNN models.
    %
    % Parameters:
    %   channel1_index : Index of the first channel (target channel)
    %   channel2_index : Index of the second channel
    %   testData       : Test data matrix [channels x samples]
    %   uniModelFile   : Filename of the univariate model file (e.g., 'univariate_model.mat')
    %   biModelFile    : Filename of the bivariate model file (e.g., 'bivariate_model.mat')
    %
    % Returns:
    %   logRatio       : Logarithmic ratio of error variances (univariate vs bivariate)
    
    % Load Models
    load('univariate_model.mat', 'model_uni');
    load('bivariate_model.mat', 'model_bi');

    % Extract data for specified channels
    channel1Data = testData(channel1_index, :);  
    channel2Data = testData(channel2_index, :);

    % Reshape for Univariate Model Prediction
    XTest_uni = reshape(channel1Data, [], 1, 1);

    % Reshape for Bivariate Model Prediction
    XTest_bi = reshape([channel1Data; channel2Data]', [], 2, 1);

    % Predict Using Univariate Model
    YPred_uni = predict(model_uni, XTest_uni);
    error_uni = channel1Data - YPred_uni;
    var_uni = var(error_uni);

    % Predict Using Bivariate Model
    YPred_bi = predict(model_bi, XTest_bi);
    error_bi = channel1Data - YPred_bi;
    var_bi = var(error_bi);

    % Compute Log Ratio
    logRatio = log(var_uni / var_bi);
    disp(['Log Ratio: ', num2str(logRatio)]);
end
