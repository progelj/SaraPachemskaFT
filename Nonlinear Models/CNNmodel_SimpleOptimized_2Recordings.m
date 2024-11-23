function logRatio = CNNmodel_SimpleOptimized_2Recordings(channel1_index_recording1, channel2_index_recording1, channel1_index_recording2, channel2_index_recording2, data_recording1, data_recording2)
    % Simple NN implementation to compute and minimize univariate error using bivariate model.
    %
    % Parameters:
    %   channel1_index_recording1 : Index of the first channel from the first
    %   recording
    %   channel2_index_recording1 : Index of the second channel from the first
    %   recording
    %   channel1_index_recording2 : Index of the first channel from the
    %   second recording
    %   channel2_index_recording2 : Index of the second channel from the
    %   second recording
    %   data_recording1           : EEG data matrix for first recording
    %   data_recording2           : EEG data matrix for second recording
    %
    % Returns:
    %   logRatio : Logarithmic ratio of error variances (univariate vs bivariate)
    
    % Extract channels data
    channel1_Rec1 = data_recording1(channel1_index_recording1, :);  
    channel2_Rec1 = data_recording1(channel2_index_recording1, :); 
    channel1_Rec2 = data_recording2(channel1_index_recording2, :);  
    channel2_Rec2 = data_recording2(channel2_index_recording2, :); 

    filterSize = 16;
    numOfFilters = 128;

       
    % Train 2 same channels from different recordings
    % Prepare bivariate input and target
    X_bi = [channel1_Rec1; channel1_Rec2];  % Inputs: channel1 and channel2
    Y_bi = channel1_Rec1;                  % Target: channel1

    % Reshape data for NN input
    XTrain_bi = reshape(X_bi', [], 2, 1);  
    YTrain_bi = reshape(Y_bi, [], 1);  


    % Define bivariate network
    layers_bi = [
        sequenceInputLayer(2)                             
        convolution1dLayer(filterSize, numOfFilters * 2, 'Padding', 'same');
        reluLayer
        fullyConnectedLayer(1)                             
    ];

    options_bi = trainingOptions('adam', ...
        'MaxEpochs', 300, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.0001, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false,...
        'ValidationData', {XVal_bi, YVal_bi});

    % Train the bivariate model
    model_bi = trainnet(XTrain_bi, YTrain_bi, layers_bi, "mse", options_bi);

    save('univariate_model_2Recordings.mat', 'model_bi');

    % Predict and compute error
    YPred_bi = predict(model_bi, XTrain_bi);
    error_bi = YTrain_bi - YPred_bi;
    var_bi = var(error_bi);  
    mse_bi = mean(error_bi.^2);

    % Train 4 channels (2 pairs) from 2 different recordings
    % Prepare the data
    X_qua = [channel1_Rec1; channel2_Rec1; channel1_Rec2; channel2_Rec2];  
    Y_qua = channel1_Rec1;        

    % Reshape data for NN input
    XTrain_qua = reshape(X_qua', [], 4, 1);  
    YTrain_qua = reshape(Y_qua, [], 1);  


    % Define bivariate network
    layers_qua = [
        sequenceInputLayer(4)                             
        convolution1dLayer(filterSize, numOfFilters * 2, 'Padding', 'same');
        reluLayer
        fullyConnectedLayer(1)                             
    ];

    options_qua = trainingOptions('adam', ...
        'MaxEpochs', 300, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.0001, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false);

    % Train the bivariate model
    model_qua = trainnet(XTrain_qua, YTrain_qua, layers_qua, "mse", options_qua);

    % save('bivariate_model.mat', 'model_bi');
    save('bivariate_model_2Recordings', 'model_qua');

    % Predict and compute error
    YPred_qua = predict(model_qua, XTrain_qua);
    error_qua = YTrain_qua - YPred_qua;
    var_qua = var(error_qua);  
    mse_qua = mean(error_qua.^2);


    % Log Ratio 
    logRatio = log(var_bi / var_qua);

    % Display the variance results
    fprintf('Variance of Univariate Error from 2 different Recordings: %.4f\n', var_bi);
    fprintf('Variance of Bivariate Error from 2 different Recordings: %.4f\n', var_qua);
    fprintf('Log Ratio: %.4f\n', logRatio);

    % Check improvement
    if var_qua < var_bi
        disp("Quadrivariate model successfully minimized the error.");
    else
        disp("No improvement ");
    end
end

