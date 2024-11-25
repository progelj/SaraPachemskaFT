function testSavedModels(channel1_index, channel2_index, dataTest)
    % Test Saved Models on New Data
    %
    % Parameters:
    %   channel1_index : Index of the first channel (target channel)
    %   channel2_index : Index of the second channel
    %   dataTest       : EEG testing data matrix
    %   modelPath      : Path to the saved models
    %
    % Returns:
    %   None

    % Extract test data
    channel1Test = dataTest(channel1_index, :);  
    channel2Test = dataTest(channel2_index, :); 

    % Load saved models
    load('univariate_model.mat', 'model_uni');
    load('bivariate_model.mat', 'model_bi');

    % Prepare data for prediction
    XVal_uni = reshape(channel1Test, [], 1, 1);
    YVal_uni = reshape(channel1Test, [], 1);
    XVal_bi = [channel1Test; channel2Test]';
    XVal_bi = reshape(XVal_bi, [], 2, 1);
    YVal_bi = reshape(channel1Test, [], 1);

    % Predict with univariate model
    YPred_uni = predict(model_uni, XVal_uni);
    error_uni = YVal_uni - YPred_uni;
    var_uni = var(error_uni);

    % Predict with bivariate model
    YPred_bi = predict(model_bi, XVal_bi);
    error_bi = YVal_bi - YPred_bi;
    var_bi = var(error_bi);

    % Display results
    fprintf('Testing Results:\n');
    fprintf('Variance of Univariate Error: %.4f\n', var_uni);
    fprintf('Variance of Bivariate Error: %.4f\n', var_bi);
    logRatio = log(var_uni / var_bi);
    fprintf('Log Ratio: %.4f\n', logRatio);
end
