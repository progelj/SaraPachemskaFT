function CNNmodel_Bivariate_simple(channel1_index, channel2_index, data)
    % CNN Model - Bivariate

    % Quick test: CNNmodel_Bivariate_simple(1, 2, EEG.data)

    % This model uses two channels for prediction
    % Input: 
    % - channel1_index: index of the first EEG channel (electrode) to use for prediction
    % - channel2_index: index of the second EEG channel (electrode) to use for prediction
    % - data: EEG data with multiple channels (rows are channels)

    % Extract data for both specified channels
    channel1Data = data(channel1_index, :);  % First channel to predict
    channel2Data = data(channel2_index, :);  % Second channel to assist prediction

    % Split data into training (80%) and testing (20%) sets
    train_ratio = 0.8;
    train_size = floor(train_ratio * (length(channel1Data) - 1));  

    % Prepare training and testing data
    % Create a 2D array for training input using both channels
    % Prepare training and testing data
    XTrain = [channel1Data(1, 1:train_size); channel2Data(1, 1:train_size)];  % Training input from both channels
    YTrain = channel1Data(1, 1:train_size);  % Training target 
    
    XTest = [channel1Data(1, train_size + 1:end); channel2Data(1, train_size + 1:end)];  % Testing input from both channels
    YTest = channel1Data(1, train_size + 1:end);  % Testing target


    % Reshape the data for CNN input (2 features per time step)
    XTrain = reshape(XTrain', [], 2, 1);    % [XTrain', numTimeSteps, numFeatures, numObservations]
    YTrain = reshape(YTrain, [], 1);        % [YTrain, numTimeSteps, 1]
    XTest = reshape(XTest', [], 2, 1);   % [XTest', numTimeSteps, numFeatures, numObservations]
    YTest = reshape(YTest, [], 1);       % [YTest, numTimeSteps, 1]


    % CNN architecture
    inputSize = 2; % Two features (channels) per time step
    numFilters = 32; % Number of filters / kernels
    filterSize = 16; % Filter size 

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
        'MaxEpochs', 200, ...               % Number of epochs
        'MiniBatchSize', 64, ...            % Mini-batch size
        'InitialLearnRate', 0.01, ...      % Learning rate
        'Shuffle', 'every-epoch', ...       % Shuffle the data every epoch
        'ValidationData', {XTest, YTest}, ... % Validation data for early stopping
        'ValidationFrequency', 10, ...
        'Plots', 'training-progress', ...   % Plot training progress
        'Verbose', false, ...                  
        'ValidationPatience', 5);          % Early stopping patience

    % Train the model using trainNetwork
    model = trainnet(XTrain, YTrain, layers, "mse", options);

    % Test the model on the test data
    YPred = predict(model, XTest);

    % Plot the predicted vs actual values for evaluation
    figure;
    plot(YPred, 'r'); % Predicted values in red
    hold on;
    plot(YTest, 'b'); % Actual values in blue
    legend('Predicted', 'Actual');
    title(['Simple CNN - Bivariate - Predicted vs Actual for Channels ' num2str(channel1_index) ' and ' num2str(channel2_index)]);

    % Evaluate the performance (Mean Squared Error)
    mseError = mean((YPred - YTest).^2);
    disp(['CNN model - Mean Squared Error on Test Data: ', num2str(mseError)]);

    % impulse_response_CNN(model);
end

function impulse_response_CNN(model)
    % Impulse signal
    impulse = zeros(100, 2); % Length of the response and two channels
    impulse(1, 1) = 1; % Unit impulse for channel 1
    impulse(1, 2) = 0; % No impulse for channel 2

    % Reshape the impulse for CNN input: [numTimeSteps, numFeatures, numObservations]
    impulse = reshape(impulse, [], 1, 2); % [numTimeSteps, 1, numFeatures]

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
