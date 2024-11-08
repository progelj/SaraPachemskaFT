function CNNmodel_Bivariate_dilationFactor(channel1_index, channel2_index,  data)
    % CNN Model with Dilated Convolutions and Residual Connections
    % Code: https://www.mathworks.com/help/deeplearning/ug/sequence-to-sequence-classification-using-1-d-convolutions.html

    % Quick test: CNNmodel_Bivariate_dilationFactor(1, 2, EEG.data)

    % This model uses two channels for prediction
    % Input: 
    % - channel1_index: index of the first EEG channel (electrode) to use for prediction
    % - channel2_index: index of the second EEG channel (electrode) to use for prediction
    % - data: EEG data with multiple channels (rows are channels)

    % Hardcoded hyperparameters
    numBlocks = 4;       % Number of residual blocks
    numFilters = 32;     % Number of filters in each convolution layer
    filterSize = 5;      % Size of the convolution filters
    dropoutFactor = 0.5; % Dropout probability for dropout layer

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
    title(['CNN MODEL Bivariate - Predicted vs Actual for Channels ' num2str(channel1_index) ' and ' num2str(channel2_index)]);

    % Evaluate the performance (Mean Squared Error)
    mseError = mean((YPred - YTest).^2);
    disp(['Dilated CNN - Bivariate - Mean Squared Error on Test Data: ', num2str(mseError)]);

    %impulse_response_CNN(model);

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



