function logRatio = testModels_AllPairs(channel1_index, channel2_index, dataTest, modelDir)
    % Test the saved models on a new dataset and compute the log ratio
    
    % Extract channel data
    channel1Test = dataTest(channel1_index, :);  
    channel2Test = dataTest(channel2_index, :); 

    % Load univariate model and test
    uniModelPath = fullfile(modelDir, sprintf('univariate_model_ch%d_ch%d.mat', channel1_index, channel2_index));
    load(uniModelPath, 'model_uni');

    XVal_uni = reshape(channel1Test, [], 1, 1);  
    YVal_uni = reshape(channel1Test, [], 1);

    YPred_uni = predict(model_uni, XVal_uni);
    error_uni = YVal_uni - YPred_uni;
    var_uni = var(error_uni);

    % Load bivariate model and test
    biModelPath = fullfile(modelDir, sprintf('bivariate_model_ch%d_ch%d.mat', channel1_index, channel2_index));
    load(biModelPath, 'model_bi');

    XVal_bi = [channel1Test; channel2Test]';  
    YVal_bi = channel1Test;

    XVal_bi = reshape(XVal_bi, [], 2, 1);  
    YVal_bi = reshape(YVal_bi, [], 1);

    YPred_bi = predict(model_bi, XVal_bi);
    error_bi = YVal_bi - YPred_bi;
    var_bi = var(error_bi);

    % Compute log ratio
    logRatio = log(var_uni / var_bi);
    fprintf('Log Ratio for channels (%d, %d): %.4f\n', channel1_index, channel2_index, logRatio);
end
