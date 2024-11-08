function logRatio = CNNmodel_Simple_FullData(channel1_index, channel2_index, data)
    % CNN Model - Bivariate with Logarithmic Error Variance Ratio
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

    % Prepare the training and testing data (using the entire data set for both)
    X = [channel1Data; channel2Data];  % Input from both channels
    Y = channel1Data;                  % Target is the first channel

    % Reshape the data for CNN input (2 features per time step)
    XTrain = reshape(X', [], 2, 1);    % [X', numTimeSteps, numFeatures, numObservations]
    YTrain = reshape(Y, [], 1);        % [Y, numTimeSteps, 1]
    XTest = XTrain;                    % Test data is the same as training data
    YTest = YTrain;                    % Test target is the same as training target

    % CNN architecture
    numFilters = 32; % Number of filters / kernels
    filterSize = 16; % Filter size 

    layers = [
        sequenceInputLayer(2)                     
        convolution1dLayer(filterSize, numFilters, 'Padding', 'same')  
        reluLayer                                         
        fullyConnectedLayer(50)                           
        reluLayer                                         
        fullyConnectedLayer(1)                            
    ];                                             

    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 200, ...               
        'MiniBatchSize', 64, ...            
        'InitialLearnRate', 0.01, ...       
        'Shuffle', 'every-epoch', ...       
        'ValidationData', {XTest, YTest}, ...
        'ValidationFrequency', 10, ... 
        'Verbose', false, ...               
        'ValidationPatience', 5);          

    % Train the bivariate model
    model = trainnet(XTrain, YTrain, layers,"mse", options);

    % Test the model on the test data
    YPred_bi = predict(model, XTest); 

    % --- Univariate model (use only the first channel for prediction) ---
    X_uni = channel1Data;  % Only the first channel as input
    Y_uni = channel1Data;  % Target is the first channel

    % Reshape data for CNN input (1 feature per time step for univariate model)
    XTrain_uni = reshape(X_uni, [], 1, 1);  
    YTrain_uni = reshape(Y_uni, [], 1);      

    layers_uni = [
        sequenceInputLayer(1)                              % Input layer with 1 feature (univariate)
        convolution1dLayer(filterSize, numFilters, 'Padding', 'same')  
        reluLayer                                          % ReLU activation layer
        fullyConnectedLayer(50)                            % Fully connected layer (50 neurons)
        reluLayer                                          % ReLU activation layer
        fullyConnectedLayer(1)                             % Output layer: predict next value
    ];  


    % Train univariate model
    model_uni = trainnet(XTrain_uni, YTrain_uni, layers_uni, "mse", options);

    % Test the univariate model
    YPred_uni = predict(model_uni, XTrain); 

    % --- Compute error variances ---
    error_bi = YTest - YPred_bi;  % Error for bivariate model
    error_uni = YTest_uni - YPred_uni;  % Error for univariate model

    % Compute the variance of errors for both models
    var_bi = var(error_bi);
    var_uni = var(error_uni);
    % 
    % % Compute the logarithmic ratio of error variances 
    logRatio = log(var_uni / var_bi);  % Logarithmic ratio of variances
end