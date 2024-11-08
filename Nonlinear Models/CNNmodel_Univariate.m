function [YPred_uni, error_uni] = CNNmodel_Univariate(channel1Data, options, filterSize, numFilters, XTest, YTest)
    % CNN Model - Univariate
    % Trains and tests the univariate model using only the first EEG channel.
    %
    % Parameters:
    %   channel1Data : array, EEG data for the first channel (input and target)
    %   options       : struct, training options (passed from the main function)
    %   filterSize    : int, filter size for convolution
    %   numFilters    : int, number of filters for the convolution layer
    %   XTest         : array, test data (input)
    %   YTest         : array, test data (target)
    %
    % Returns:
    %   YPred_uni    : array, predictions from the univariate model
    %   error_uni    : array, prediction errors for univariate model

    % Prepare the training and testing data (univariate)
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

    % Train the univariate model
    model_uni = trainnet(XTrain_uni, YTrain_uni, layers_uni, "mse", options);

    % Test the univariate model
    YPred_uni = predict(model_uni, XTest); 

    % Compute error for univariate model
    error_uni = YTest - YPred_uni;  % Error for univariate model

end
