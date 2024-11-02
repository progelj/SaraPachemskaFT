function CNNmodel_Univariate_initial(channel_index, data)
    % CNN Model - Univariate

    % Quick test: CNNmodel_Univariate_initial(1, EEG.data)

    % Input: 
    % - channel_index: index of the EEG channel (electrode) to use for prediction
    % - EEG_data: EEG data

    % Impulse response

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
    filterSize = 16; % Filter size 

    layers = [
        sequenceInputLayer(inputSize)               % Sequence input with 1 feature per time step
        convolution1dLayer(filterSize, numFilters, 'Padding', 'same') % 1D convolutional layer
        batchNormalizationLayer                     % Batch normalization layer
        reluLayer                                   % ReLU activation layer
        dropoutLayer(0.2)                           % 20% of the neurons are randomly dropped

        convolution1dLayer(filterSize, numFilters*2, 'Padding', 'same')  % Second convolutional layer
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.2)

        fullyConnectedLayer(100)                    % Fully connected layer (100 neurons)
        reluLayer                                   % ReLU activation layer

        fullyConnectedLayer(1)                      % Output layer: predict next value
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

    % layerFileName = 'CNN_Layers.mat'; % Specify the file name
    % save(layerFileName, 'layers'); % Save the layers variable
    % disp(['Layers saved to ', layerFileName]);

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
    title(['CNN MODEL - Predicted vs Actual for Channel ' num2str(channel_index)]);

    % Evaluate the performance (Mean Squared Error)
    mseError = mean((YPred - YTest).^2);
    disp(['CNN model - Mean Squared Error on Test Data: ', num2str(mseError)]);

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