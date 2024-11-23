function logRatio = LoadAndTestModelCNN_2Recordings(channel1_index_recording1, channel2_index_recording1,testData)
    % computeLogRatio: Compute the log ratio of error variances using trained CNN models.
    %
    % Parameters:
    %   channel1_index : Index of the first channel (target channel)
    %   channel2_index : Index of the second channel
    %   testData       : Test data matrix [channels x samples]
    %
    % Returns:
    %   logRatio       : Logarithmic ratio of error variances (univariate vs bivariate)
    
    % Load Models
    load('univariate_model_2Recordings.mat', 'model_bi');
    load('bivariate_model_2Recordings.mat', 'model_qua');

    % Extract data for specified channels
    channel1_Rec1 = testData(channel1_index_recording1, :);  
    channel2_Rec1 = testData(channel2_index_recording1, :); 

    % Reshape for first (bi) Model Prediction
    XTest_bi = reshape([channel1_Rec1; channel2_Rec1]', [], 2, 1);

    % Reshape for second(qua) Model Prediction
    XTest_qua = reshape([channel1_Rec1; channel2Data]', [], 2, 1);

    % Predict 
    YPred_bi = predict(model_bi, XTest_bi);
    error_uni = channel1_Rec1 - YPred_uni;
    var_uni = var(error_uni);

    

    % Compute Log Ratio
    logRatio = log(var_uni / var_bi);
    disp(['Variance of Univariate Error: ', num2str(var_uni)]);
    disp(['Variance of Bivariate Error: ', num2str(var_bi)]);
    disp(['Log Ratio: ', num2str(logRatio)]);
end
