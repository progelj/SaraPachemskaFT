function CNNmodel_Univariate_dilationFactor(channel_index, data)
    % CNN Model with Dilated Convolutions and Residual Connections
    % Code: https://www.mathworks.com/help/deeplearning/ug/sequence-to-sequence-classification-using-1-d-convolutions.html

    % Quick test: CNNmodel_Univariate_dilationFactor(1, EEG.data)

    % Input: 
    % - channel_index: index of the EEG channel (electrode) to use for prediction
    % - data: EEG data

    % Hardcoded hyperparameters
    numBlocks = 4;       % Number of residual blocks
    numFilters = 32;     % Number of filters in each convolution layer
    filterSize = 5;      % Size of the convolution filters
    dropoutFactor = 0.5; % Dropout probability for dropout layer

    % Extract the channel data 
    inputData = data(channel_index, :); 

    % Split data into training (80%) and testing (20%) sets
    train_ratio = 0.8;
    train_size = floor(train_ratio * (length(inputData) - 1));  

    % Prepare training and testing data
    XTrain = inputData(1, 1:train_size);  % Training input
    YTrain = inputData(1, 1:train_size);  % Training target 

    XTest = inputData(train_size + 1:end);  % Testing input
    YTest = inputData(train_size + 1:end);  % Testing target 

    % Reshape the data for CNN input (1 feature per time step)
    XTrain = reshape(XTrain, [], 1, 1);   % [numTimeSteps, 1, 1] for sequenceInputLayer [numberTimeSteps, numberFeatures, numberObservations]
    YTrain = reshape(YTrain, [], 1);      % [numTimeSteps, 1] (1 for one column) 
    XTest = reshape(XTest, [], 1, 1);     % [numTimeSteps, 1, 1]
    YTest = reshape(YTest, [], 1);        % [numTimeSteps, 1]

    % Initialize the dlnetwork
    net = dlnetwork;

    % Input layer
    layer = sequenceInputLayer(1,Normalization="rescale-symmetric",Name="input");
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
        'InitialLearnRate', 0.1, ...       % Learning rate
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
    title(['CNN MODEL with Dilated Convolution - Predicted vs Actual for Channel ' num2str(channel_index)]);

    % Evaluate the performance (Mean Squared Error)
    mseError = mean((YPred - YTest).^2);
    disp(['Dilated CNN - Mean Squared Error on Test Data: ', num2str(mseError)]);

    impulse_response_CNN(model);

end



function impulse_response_CNN(model)
    % 
    % Impulse signal
    impulse = zeros(100, 1); % Length of the response
    impulse(1) = 1; % Unit impulse

    % Reshape the impulse for CNN input: [numTimeSteps, numFeatures, numObservations]
    impulse = reshape(impulse, [], 1, 1); 

    % Predict using the CNN model
    impulse_response = predict(model, impulse);
    
    % Check if the impulse response needs reshaping
    if size(impulse_response, 1) > 1
        impulse_response = squeeze(impulse_response); 
    end

    % Plot 
    figure;
    plot(impulse_response, 'LineWidth', 2);
    title('Impulse Response of CNN Model');
    xlabel('Samples');
    ylabel('Amplitude');
end



