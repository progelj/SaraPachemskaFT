function CNNmodel_Univariate_dilationFactor(channel_index, data)
    % CNN Model with Dilated Convolutions and Residual Connections
    % Code: https://www.mathworks.com/help/deeplearning/ug/sequence-to-sequence-classification-using-1-d-convolutions.html

    % Quick test: CNNmodel_Univariate_dilationFactor(1, EEG.data)

    % Input: 
    % - channel_index: index of the EEG channel (electrode) to use for prediction
    % - data: EEG data

    % Hardcoded hyperparameters
    numBlocks = 4;        % Number of residual blocks
    numFilters = 32;      % Number of filters in each convolution layer
    filterSize = 16;      % Size of the convolution filters
    dropoutFactor = 0.5;  % Dropout probability for dropout layer

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
        'ValidationPatience', 5);          % Early stopping patience

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

    impulse_response_CNN(model, filterSize);

end



function impulse_response_CNN(model, filterSize)
   % Initialize impulse signal with a single unit impulse
    impulse_length = 100;  % Length of the impulse response
    impulse_response = zeros(impulse_length, 1);  
    impulse_window = zeros(filterSize, 1); 
    impulse_window(1) = 1;  % Set the unit impulse in the first position

    % Calculate the impulse response 
    for t = 1:impulse_length
        if t > filterSize
            impulse_response(t) = 0;
        else
            % Reshape [filterSize, 1, 1]
            input_window = reshape(impulse_window, [filterSize, 1, 1]);

            % Predict the next value
            next_value = predict(model, input_window);

            % Ensure next_value is scalar
            next_value = squeeze(next_value);

            % Confirm that next_value is scalar; if not, take only the first element
            if numel(next_value) > 1
                next_value = next_value(1);
            end

            % Store the prediction in the response vector
            impulse_response(t) = next_value;

            % Slide the window forward: shift and add the new prediction as the last element
            impulse_window = [impulse_window(2:end); next_value];
        end
    end

    % Plot the impulse response
    figure;
    plot(impulse_response, 'LineWidth', 2);
    title('Autoregressive Impulse Response of CNN Model');
    xlabel('Samples');
    ylabel('Amplitude');
end



