function CNNmodel_Bivariate_initial_FullData(channel1_index, channel2_index, data)
    % CNN Model - Bivariate

    % Quick test: CNNmodel_Bivariate_initial_FullData(1, 2, EEG.data)

    % This model uses two channels for prediction and uses full dataset
    % Input: 
    % - channel1_index: index of the first EEG channel (electrode) to use for prediction
    % - channel2_index: index of the second EEG channel (electrode) to use for prediction
    % - data: EEG data with multiple channels (rows eeare channels)

    % Extract the channel data (single channel for the specified index)
    channel1Data = data(channel1_index, :);  % First channel to predict
    channel2Data = data(channel2_index, :);  % Second channel to assist prediction

    % Use the entire dataset for both training and testing
    X = [channel1Data; channel2Data];  % Input from both channels
    Y = channel1Data;                  % Target is the first channel

    % Reshape the data for CNN input (2 features per time step)
    XTrain = reshape(X', [], 2, 1);    % [X', numTimeSteps, numFeatures, numObservations]
    YTrain = reshape(Y, [], 1);        % [Y, numTimeSteps, 1]
    XTest = XTrain;                    % Test data is the same as training data
    YTest = YTrain;                    % Test target is the same as training target

    % CNN architecture
    inputSize = 2; % Two features (channels) per time step
    numFilters = 32; % Number of filters / kernels
    filterSize = 10; % Filter size 

    layers = [
        sequenceInputLayer(inputSize)                     % Sequence input with multiple features (channels)
        convolution1dLayer(filterSize, numFilters, 'Padding', 'same')  % 1D convolutional layer
        batchNormalizationLayer                           % Batch normalization layer
        reluLayer                                         % ReLU activation layer
        dropoutLayer(0.2)                                 % 20% dropout

        convolution1dLayer(filterSize, numFilters * 2, 'Padding', 'same') % Second convolutional layer
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.2)

        fullyConnectedLayer(100)                          % Fully connected layer (100 neurons)
        reluLayer                                         % ReLU activation layer
        fullyConnectedLayer(1)                            % Output layer: predict next value
    ];                                             

    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 200, ...               % Number of epochs
        'MiniBatchSize', 64, ...            % Mini-batch size
        'InitialLearnRate', 0.001, ...      % Learning rate
        'Shuffle', 'every-epoch', ...       % Shuffle the data every epoch
        'ValidationData', {XTest, YTest}, ... % Validation data for early stopping
        'ValidationFrequency', 10, ...
        'Plots', 'training-progress', ...   % Plot training progress
        'Verbose', false, ...                   
        'ValidationPatience', 5);          % Early stopping patience

    % Train the CNN model using trainnet and 'mse' loss function
    model = trainnet(XTrain, YTrain, layers, "mse", options);

    % Test the model on the test data
    YPred = predict(model, XTest);

    % Plot the predicted vs actual values for evaluation
    figure;
    plot(YPred, 'r'); % Predicted values in red
    hold on;
    plot(YTest, 'b'); % Actual values in blue
    legend('Predicted', 'Actual');
    title(['Initial CNN - Bivariate - Predicted vs Actual for Channels ' num2str(channel1_index) ' and ' num2str(channel2_index)]);

    % Evaluate the performance (Mean Squared Error)
    mseError = mean((YPred - YTest).^2);
    disp(['Initial CNN model - Mean Squared Error on Full Dataset: ', num2str(mseError)]);

  
end

