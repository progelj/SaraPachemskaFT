function logRatio = CNNmodel_DifferentCNNs_FullData(channel1_index, channel2_index, data)
    % CNN Model - Simple Solution
    % Computes the log ratio of error variances for a CNN-based model using two channels.
    %
    % Parameters:
    %   channel1_index : int, index of the first EEG channel
    %   channel2_index : int, index of the second EEG channel
    %   data           : matrix, EEG data (rows are channels, columns are time points)
    %
    % Returns:
    %   logRatio       : float, logarithmic error variance ratio for the two channels

    % Extract data for both specified channels
    channel1Data = data(channel1_index, :);  % First channel to predict
    channel2Data = data(channel2_index, :);  % Second channel to assist prediction

    % --- Bivariate model ---

    % Prepare the training and testing data (using the entire data set for both)
    X_bi = [channel1Data; channel2Data];  % Input from both channels
    Y_bi = channel1Data;                  % Target is the first channel

    % Reshape the data for CNN input (2 features per time step)
    XTrain_bi = reshape(X_bi', [], 2, 1);    % [X', numTimeSteps, numFeatures, numObservations]
    YTrain_bi = reshape(Y_bi, [], 1);        % [Y, numTimeSteps, 1]
    XTest_bi = XTrain_bi;                    % Test data is the same as training data
    YTest_bi = YTrain_bi;                    % Test target is the same as training target
   
    % CNN architecture parameters
    numFilters = 32; % Number of filters / kernels
    filterSize = 5; % Filter size 

    layers_bi = [
        sequenceInputLayer(2)                     
        convolution1dLayer(filterSize, numFilters, 'Padding', 'same') % 1D convolutional layer
        batchNormalizationLayer                                       % Batch normalization for regularization
        reluLayer                                                     % ReLU activation


        convolution1dLayer(filterSize, numFilters, 'Padding', 'same') % 1D convolutional layer
        batchNormalizationLayer                                       % Batch normalization for regularization
        reluLayer                                                     % ReLU activation
        
        
        convolution1dLayer(filterSize, numFilters * 2, 'Padding', 'same')  % Second convolutional layer with more filters
        batchNormalizationLayer                                       % Batch normalization for regularization
        reluLayer                                                     % ReLU activation

        convolution1dLayer(filterSize, numFilters * 4, 'Padding', 'same')  % Second convolutional layer with more filters
        batchNormalizationLayer                                       % Batch normalization for regularization
        reluLayer                                                     % ReLU activation
        
        dropoutLayer(0.5)                                             % Dropout layer for regularization
       
        fullyConnectedLayer(100)                                      % Fully connected layer
        reluLayer                                                     % ReLU activation
        
        fullyConnectedLayer(1)                                        % Output layer                          
    ];                                             

    % Training options
    options_bi = trainingOptions('adam', ...
        'MaxEpochs', 300, ...               
        'MiniBatchSize', 64, ...            
        'InitialLearnRate', 0.0001, ...       
        'Shuffle', 'every-epoch', ...       
        'ValidationData', {XTest_bi, YTest_bi}, ...
        'ValidationFrequency', 10, ... 
        'Verbose', false, ...               
        'ValidationPatience', 10);          

    % Train the bivariate model
    model_bi = trainnet(XTrain_bi, YTrain_bi, layers_bi,"mse", options_bi);

    % Test the model - Prediction for the bivariate model
    YPred_bi = predict(model_bi, XTest_bi); 

    % --- Univariate model ---
    X_uni = channel1Data;  % Only the first channel as input
    Y_uni = channel1Data;  % Target is the first channel

    % Reshape data for CNN input (1 feature per time step for univariate model)
    XTrain_uni = reshape(X_uni, [], 1, 1);  
    YTrain_uni = reshape(Y_uni, [], 1);      
    Xtest_uni = XTrain_uni;
    YTest_uni = YTrain_uni;

    layers_uni = [
        sequenceInputLayer(1)                              % Input layer with 1 feature (univariate)
        convolution1dLayer(filterSize, numFilters, 'Padding', 'same') % 1D convolutional layer
        batchNormalizationLayer                                       % Batch normalization for regularization
        reluLayer                                                     % ReLU activation

        convolution1dLayer(filterSize, numFilters, 'Padding', 'same') % 1D convolutional layer
        batchNormalizationLayer                                       % Batch normalization for regularization
        reluLayer                                                     % ReLU activation
        
        convolution1dLayer(filterSize, numFilters * 2, 'Padding', 'same')  % Second convolutional layer with more filters
        batchNormalizationLayer                                       % Batch normalization for regularization
        reluLayer                                                     % ReLU activation

        convolution1dLayer(filterSize, numFilters, 'Padding', 'same') % 1D convolutional layer
        batchNormalizationLayer                                       % Batch normalization for regularization
        reluLayer   

        convolution1dLayer(filterSize, numFilters, 'Padding', 'same') % 1D convolutional layer
        batchNormalizationLayer                                       % Batch normalization for regularization
        reluLayer  

        fullyConnectedLayer(100)                                      % Fully connected layer    
        reluLayer                                                     % ReLU activation
        dropoutLayer(0.5)                                             % Dropout layer for regularization
        fullyConnectedLayer(1)                                        % Output layer                                         
    ];  

     options_uni = trainingOptions('adam', ...
        'MaxEpochs', 300, ...               
        'MiniBatchSize', 64, ...            
        'InitialLearnRate', 0.0001, ...       
        'Shuffle', 'every-epoch', ...       
        'ValidationData', {Xtest_uni, YTest_uni}, ...
        'ValidationFrequency', 10, ... 
        'Verbose', false, ...               
        'ValidationPatience', 10);     

    % Train univariate model
    model_uni = trainnet(XTrain_uni, YTrain_uni, layers_uni, "mse", options_uni);

    % Test the univariate model
    YPred_uni = predict(model_uni, Xtest_uni); 

    % --- Compute error variances ---
    error_bi = YTest_bi - YPred_bi;  % Error for bivariate model
    error_uni = YTest_uni - YPred_uni;  % Error for univariate model

    % Compute the variance of errors for both models
    var_bi = var(error_bi);
    var_uni = var(error_uni);
    % 
    % % Compute the logarithmic ratio of error variances 
    logRatio = log(var_uni / var_bi);  % Logarithmic ratio of variances

    disp("LogRatio:" + logRatio);
end