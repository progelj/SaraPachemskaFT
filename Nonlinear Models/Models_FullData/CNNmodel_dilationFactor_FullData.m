function logRatio = CNNmodel_dilationFactor_FullData(channel1_index, channel2_index, data)
    % CNN Model with Dilated Convolutions and Residual Connections
    % Computes the log ratio of error variances for a CNN-based model using two channels.
    %
    % Parameters:
    %   channel1_index : int, index of the first EEG channel
    %   channel2_index : int, index of the second EEG channel
    %   data           : matrix, EEG data (rows are channels, columns are time points)
    %
    % Returns:
    %   logRatio       : float, logarithmic error variance ratio for the two channels

    % Hyperparameters
    numBlocks = 4;       % Number of residual blocks
    numFilters = 32;     % Number of filters in each convolution layer
    filterSize = 3;      % Size of the convolution filters
    dropoutFactor = 0.005; % Dropout probability for dropout layer

    % Extract data for both specified channels
    channel1Data = data(channel1_index, :);  % First channel to predict
    channel2Data = data(channel2_index, :);  % Second channel to assist prediction


    % --- Univariate model (using only the first channel) ---

    % Prepare data
    X_uni = channel1Data;  % Only the first channel as input
    Y_uni = channel1Data;  % Target is the first channel

    % Reshape for CNN input (1 feature per time step)
    XTrain_uni = reshape(X_uni, [], 1, 1);  
    YTrain_uni = reshape(Y_uni, [], 1);      
    XTest_uni = XTrain_uni;                  
    YTest_uni = YTrain_uni;                  

    % CNN architecture for univariate data
    net_uni = createDilatedResNet(numBlocks, numFilters, filterSize, dropoutFactor, 1);

    options_uni = trainingOptions('adam', ...
        'MaxEpochs', 200, ...               
        'MiniBatchSize', 64, ...            
        'InitialLearnRate', 0.1, ...        
        'Shuffle', 'every-epoch', ...       
        'ValidationData', {XTest_uni, YTest_uni}, ...
        'ValidationFrequency', 10, ...
        'Verbose', false, ...                   
        'ValidationPatience', 10);

    % Train and predict
    model_uni = trainnet(XTrain_uni, YTrain_uni, net_uni, "mse", options_uni);
    YPred_uni = predict(model_uni, XTest_uni);

    % --- Bivariate model (using both channels) ---
    % Prepare data
    X_bi = [channel1Data; channel2Data];  % Input from both channels
    Y_bi = channel1Data;                  % Target is the first channel

    % Reshape for CNN input (2 features per time step)
    XTrain_bi = reshape(X_bi', [], 2, 1);  
    YTrain_bi = reshape(Y_bi, [], 1);       
    XTest_bi = XTrain_bi;                   
    YTest_bi = YTrain_bi;                   

    % CNN for bivariate model
    net_bi = createDilatedResNet(numBlocks, numFilters, filterSize, dropoutFactor, 2);

    save('CNN_dilation.mat', 'net_bi');


    % Training options
    options_bi = trainingOptions('adam', ...
        'MaxEpochs', 200, ...               
        'MiniBatchSize', 64, ...            
        'InitialLearnRate', 0.1, ...        
        'Shuffle', 'every-epoch', ...       
        'ValidationData', {XTest_bi, YTest_bi}, ...
        'ValidationFrequency', 10, ...
        'Verbose', false, ...                   
        'ValidationPatience', 10);

    % Train and predict
    model_bi = trainnet(XTrain_bi, YTrain_bi, net_bi, "mse", options_bi);
    YPred_bi = predict(model_bi, XTest_bi);

    % --- Compute error variances ---

    % Calculate errors
    error_bi = YTest_bi - YPred_bi;  % Error for bivariate model
    error_uni = YTest_uni - YPred_uni;  % Error for univariate model

    % Compute variance of errors
    var_bi = var(error_bi);
    var_uni = var(error_uni);

    % Logarithmic ratio of variances
    logRatio = log(var_uni / var_bi);
      % Compute MSE for both models
    mse_bi = mean(error_bi.^2);  % MSE for bivariate model
    mse_uni = mean(error_uni.^2);  % MSE for univariate model

    % Display the results 
    fprintf('Bivariate Model MSE: %.4f\n', mse_bi);
    fprintf('Univariate Model MSE: %.4f\n', mse_uni);
end

function net = createDilatedResNet(numBlocks, numFilters, filterSize, dropoutFactor, inputSize)
    % Helper function to create the dilated residual CNN network
    % Parameters:
    %   numBlocks    : Number of residual blocks
    %   numFilters   : Number of filters per convolution layer
    %   filterSize   : Size of the convolution filter
    %   dropoutFactor: Dropout rate for dropout layers
    %   inputSize    : Number of input features (1 for univariate, 2 for bivariate)
    
    net = dlnetwork;
    layer = sequenceInputLayer(inputSize, Normalization="rescale-symmetric", Name="input");
    net = addLayers(net, layer);
    outputName = layer.Name;

    for i = 1:numBlocks
        dilationFactor = 2^(i-1);  % Exponential dilation factor

        % Residual block layers
        layers = [
            convolution1dLayer(filterSize, numFilters, DilationFactor=dilationFactor, Padding="causal", Name="conv1_" + i)
            layerNormalizationLayer(Name="layerNorm1_" + i)
            dropoutLayer(dropoutFactor, Name="dropout1_" + i)
            convolution1dLayer(filterSize, numFilters, DilationFactor=dilationFactor, Padding="causal", Name="conv2_" + i)
            layerNormalizationLayer(Name="layerNorm2_" + i)
            reluLayer(Name="relu_" + i)
            dropoutLayer(dropoutFactor, Name="dropout2_" + i)
            additionLayer(2, Name="add_" + i)
        ];

        net = addLayers(net, layers);
        net = connectLayers(net, outputName, "conv1_" + i);

        % Skip connection
        if i == 1
            layer = convolution1dLayer(1, numFilters, Name="convSkip");
            net = addLayers(net, layer);
            net = connectLayers(net, outputName, "convSkip");
            net = connectLayers(net, "convSkip", "add_" + i + "/in2");
        else
            net = connectLayers(net, outputName, "add_" + i + "/in2");
        end

        outputName = "add_" + i;
    end

    % Fully connected output layer
    layers = fullyConnectedLayer(1, Name="fc");
    net = addLayers(net, layers);
    net = connectLayers(net, outputName, "fc");
end
