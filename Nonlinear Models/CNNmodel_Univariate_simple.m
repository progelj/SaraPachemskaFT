function CNNmodel_Univariate_simple(channel_index, data)
    % CNN Model - Univariate
    % Simpler model - Best results

    % Quick test: CNNmodel_Univariate_simple(1, EEG.data)

    % Input: 
    % - channel_index: index of the EEG channel (electrode) to use for prediction
    % - EEG_data: EEG data

    % Extract the channel data (single channel for the specified index)
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
    YTrain = reshape(YTrain, [], 1);      % [numTimeSteps, 1] ( 1 - one column) 
    XTest = reshape(XTest, [], 1, 1);      % [numTimeSteps, 1, 1]
    YTest = reshape(YTest, [], 1);         % [numTimeSteps, 1]

    % CNN architecture
    inputSize = 1; % One feature (channel) per time step
    numFilters = 32; % Number of filters / kernels
    filterSize = 5; % Filter size 

    layers = [
        sequenceInputLayer(inputSize)                     % Sequence input with multiple features (channels)
        convolution1dLayer(filterSize, numFilters, 'Padding', 'same')  % 1D convolutional layer
        reluLayer                                         % ReLU activation layer

        fullyConnectedLayer(50)                           % Fully connected layer (50 neurons)
        reluLayer                                         % ReLU activation layer

        fullyConnectedLayer(1)                            % Output layer: predict next value
    ];                                             

    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 500, ...               % Number of epochs
        'MiniBatchSize', 64, ...            % Mini-batch size
        'InitialLearnRate', 0.01, ...      % Learning rate
        'Shuffle', 'every-epoch', ...       % Shuffle the data every epoch
        'ValidationData', {XTest, YTest}, ... % Validation data for early stopping
        'ValidationFrequency', 10, ...
        'Plots', 'training-progress', ...   % Plot training progress
        'Verbose', false, ...                  
        'ValidationPatience', 10);          % Early stopping patience

    % layerFileName = 'CNN_Layers.mat'; % Specify the file name
    % save(layerFileName, 'layers'); % Save the layers variable
    % disp(['Layers saved to ', layerFileName]);

    model = trainnet(XTrain, YTrain, layers, "mse", options);

    % Test the model on the test data
    YPred = predict(model, XTest);

    % Plot the predicted vs actual values for evaluation
    figure;
    plot(YPred, 'r'); % Predicted values in red
    hold on;
    plot(YTest, 'b'); % Actual values in blue
    legend('Predicted', 'Actual');
    title(['CNN MODEL - Predicted vs Actual for Channel ' num2str(channel_index)]);

    % Evaluate the performance (Mean Squared Error)
    mseError = mean((YPred - YTest).^2);
    disp(['CNN model - Mean Squared Error on Test Data: ', num2str(mseError)]);

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