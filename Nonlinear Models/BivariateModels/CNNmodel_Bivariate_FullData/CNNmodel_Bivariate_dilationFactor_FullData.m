function CNNmodel_Bivariate_dilationFactor_FullData(channel1_index, channel2_index, data)
    % CNN Model with Dilated Convolutions and Residual Connections

    % Quick test: CNNmodel_Bivariate_dilationFactor_FullData(1, 2, EEG.data)
    
    % This model uses two channels for prediction and the full datset

    % Hardcoded hyperparameters
    numBlocks = 4;       % Number of residual blocks
    numFilters = 32;     % Number of filters in each convolution layer
    filterSize = 5;      % Size of the convolution filters
    dropoutFactor = 0.5; % Dropout probability for dropout layer

    % Extract the channel data 
    channel1Data = data(channel1_index, :);  % First channel to predict
    channel2Data = data(channel2_index, :);  % Second channel to assist prediction

    % Use the entire dataset as training and testing data
    X = [channel1Data; channel2Data];  % Input from both channels
    Y = channel1Data;                  % Target is the first channel

    % Reshape the data for CNN input (2 features per time step)
    XTrain = reshape(X', [], 2, 1);    % [X', numTimeSteps, numFeatures, numObservations]
    YTrain = reshape(Y, [], 1);        % [Y, numTimeSteps, 1]
    XTest = XTrain;                    % Test data is the same as training data
    YTest = YTrain;                    % Test target is the same as training target

    % Initialize the dlnetwork
    net = dlnetwork;

    % Input layer
    layer = sequenceInputLayer(2, Normalization="rescale-symmetric", Name="input");
    net = addLayers(net, layer);
    outputName = layer.Name;

    % Residual blocks with dilation and skip connections
    for i = 1:numBlocks
        dilationFactor = 2^(i-1);  % Exponential increase in dilation factor

        % Residual block
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

        % Add layers to the network and connect them
        net = addLayers(net, layers);
        net = connectLayers(net, outputName, "conv1_" + i);

        % Skip connection
        if i == 1
            % Include convolution in the first skip connection
            layer = convolution1dLayer(1, numFilters, Name="convSkip");
            net = addLayers(net, layer);
            net = connectLayers(net, outputName, "convSkip");
            net = connectLayers(net, "convSkip", "add_" + i + "/in2");
        else
            net = connectLayers(net, outputName, "add_" + i + "/in2");
        end

        % Update output name for the next block
        outputName = "add_" + i;
    end

    % Fully connected output layer (regression for single value prediction)
    layers = [
        fullyConnectedLayer(1, Name="fc")
    ];
    net = addLayers(net, layers);
    net = connectLayers(net, outputName, "fc");

    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 200, ...               % Number of epochs
        'MiniBatchSize', 64, ...            % Mini-batch size
        'InitialLearnRate', 0.1, ...        % Learning rate
        'Shuffle', 'every-epoch', ...       % Shuffle the data every epoch
        'ValidationData', {XTest, YTest}, ... % Validation data for early stopping
        'ValidationFrequency', 10, ...
        'Plots', 'training-progress', ...   % Plot training progress
        'Verbose', false, ...                   
        'ValidationPatience', 10);          % Early stopping patience

    % Train the network using trainnet and MSE loss
    model = trainnet(XTrain, YTrain, net, "mse", options);

    % Test the model on the test data
    YPred = predict(model, XTest);

    % Plot the predicted vs actual values for evaluation
    figure;
    plot(YPred, 'r'); % Predicted values in red
    hold on;
    plot(YTest, 'b'); % Actual values in blue
    legend('Predicted', 'Actual');
    title(['Dilated CNN - Bivariate - Predicted vs Actual for Channels ' num2str(channel1_index) ' and ' num2str(channel2_index)]);

    % Evaluate the performance (Mean Squared Error)
    mseError = mean((YPred - YTest).^2);
    disp(['Dilated CNN model - Bivariate - Mean Squared Error on Full Dataset: ', num2str(mseError)]);

end
