function connectivityMatrix = CNNmodel_Bivariate_dilationFactor_connectivityMatrix(data)
    % Connectivity Matrix. Matrix where each element (i, j) represents the MSE
    % of the CNN model when using channel i and channel j for prediction 

    % Quick test: CNNmodel_Bivariate_dilationFactor_connectivityMatrix(EEG.data)

    % Input:
    % - data: EEG data 
    % Output:
    % - connectivityMatrix
    
    numChannels = size(data, 1); % Number of channels 
    disp(numChannels);
    connectivityMatrix = zeros(numChannels, numChannels); % Initialize the connectivity matrix

    % Parallel Computing 
    parfor ch1 = 1:numChannels
        for ch2 = 1:numChannels
            if ch1 ~= ch2
                fprintf('Processing channel pair: (%d, %d)\n', ch1, ch2);
                mseError = CNNmodel_Bivariate_test(ch1, ch2, data);
                connectivityMatrix(ch1, ch2) = mseError;
            else
                connectivityMatrix(ch1, ch2) = NaN;
            end
        end
    end

    
    % Display the connectivity matrix as a heatmap for visualization
    figure;
    imagesc(connectivityMatrix);
    colorbar;
    
    % Adjust the axes to place (1,1) in the bottom-left corner
    set(gca, 'YDir', 'normal');  % Reverses the y-axis direction so 1 is at the bottom
    
    % Set the tick labels to match the channel numbers starting from 1
    xticks(1:numChannels);
    yticks(1:numChannels);
    
    % Add titles and labels
    title('Connectivity Matrix - MSE between Channel Pairs');
    xlabel('Channel');
    ylabel('Channel');

end

function mseError = CNNmodel_Bivariate_test(channel1_index, channel2_index, data)
    % CNN Model with Dilated Convolutions and Residual Connections
    % Code: https://www.mathworks.com/help/deeplearning/ug/sequence-to-sequence-classification-using-1-d-convolutions.html
    
    % This model uses two channels for prediction

    % Hardcoded hyperparameters
    numBlocks = 4;        % Number of residual blocks
    numFilters = 32;      % Number of filters in each convolution layer
    filterSize = 16;      % Size of the convolution filters
    dropoutFactor = 0.5;  % Dropout probability for dropout layer

    % Extract the channel data 
    channel1Data = data(channel1_index, :);  % First channel to predict
    channel2Data = data(channel2_index, :);  % Second channel to assist prediction

    % Split data into training (80%) and testing (20%) sets
    train_ratio = 0.8;
    train_size = floor(train_ratio * (length(channel1Data) - 1));  

    XTrain = [channel1Data(1, 1:train_size); channel2Data(1, 1:train_size)];  % Training input from both channels
    YTrain = channel1Data(1, 1:train_size);  % Training target 
    
    XTest = [channel1Data(1, train_size + 1:end); channel2Data(1, train_size + 1:end)];  % Testing input from both channels
    YTest = channel1Data(1, train_size + 1:end);  % Testing target


    % Reshape the data for CNN input (2 features per time step)
    XTrain = reshape(XTrain', train_size, 2, 1);    % [XTrain', numTimeSteps, numFeatures, numObservations]
    YTrain = reshape(YTrain, train_size, 1);        % [YTrain, numTimeSteps, 1]
    XTest = reshape(XTest', length(XTest), 2, 1);   % [XTest', numTimeSteps, numFeatures, numObservations]
    YTest = reshape(YTest, length(YTest), 1);       % [YTest, numTimeSteps, 1]

    % Initialize the dlnetwork
    net = dlnetwork;

    % Input layer
    layer = sequenceInputLayer(2,Normalization="rescale-symmetric",Name="input");
    net = addLayers(net,layer);
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
        'Verbose', false, ...                  
        'ValidationPatience', 5);           % Early stopping patience

    % Train the network using trainnet and MSE loss
    model = trainnet(XTrain, YTrain, net, "mse", options);

    % Test the model on the test data
    YPred = predict(model, XTest);

    % Evaluate the performance (Mean Squared Error)
    mseError = mean((YPred - YTest).^2);
   
end
