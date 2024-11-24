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
    
    % Script to Test Models Trained on Two Recordings with Data from One Recording

    % Load trained models
    load('univariate_model_2Recordings.mat', 'model_bi');
    load('bivariate_model_2Recordings.mat', 'model_qua');
    
    % Load or define test data
    % Replace 'test_data_recording' with your single test recording
    % Format: EEG data matrix where rows represent channels and columns represent time samples.
   
 
    % Extract test data
    test_channel1_data = testData(channel1_index_recording1, :);
    test_channel2_data = testData(channel2_index_recording1, :);
    
    % Prepare test data for univariate model
    % Since the model was trained on two recordings, duplicate the same channel for the second recording
    XTest_bi = [test_channel1_data; test_channel1_data]; % Use the same channel twice
    XTest_bi = reshape(XTest_bi', [], 2, 1);
    
    % Predict using the univariate model
    YPred_bi = predict(model_bi, XTest_bi);
    
    % Compute error for the univariate model
    error_bi = test_channel1_data(:) - YPred_bi(:);
    var_bi = var(error_bi);
    mse_bi = mean(error_bi.^2);
    
    % Bivariate Prediction
    % Prepare test data for bivariate model
    % Duplicate the recording's channels to simulate data for two recordings
    XTest_qua = [test_channel1_data; test_channel2_data; test_channel1_data; test_channel2_data];
    XTest_qua = reshape(XTest_qua', [], 4, 1);
    
    % Predict using the bivariate model
    YPred_qua = predict(model_qua, XTest_qua);
    
    % Compute error for the bivariate model
    error_qua = test_channel1_data(:) - YPred_qua(:);
    var_qua = var(error_qua);
    mse_qua = mean(error_qua.^2);
    
    % Log Ratio and Results
    % Compute and display the log ratio of error variances
    logRatio = log(var_bi / var_qua);
    
    % Display the results
    fprintf('Test Results (Single Recording):\n');
    fprintf('Variance of Univariate Error: %.4f\n', var_bi);
    fprintf('Variance of Bivariate Error: %.4f\n', var_qua);
    fprintf('Log Ratio (Univariate/Bivariate): %.4f\n', logRatio);
    
    % Check improvement
    if var_qua < var_bi
        disp('Bivariate model successfully minimized the error on single recording test data.');
    else
        disp('No improvement in bivariate model on single recording test data.');
    end

end
