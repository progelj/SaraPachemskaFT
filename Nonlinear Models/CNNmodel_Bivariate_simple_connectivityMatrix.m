function connectivityMatrix = CNNmodel_Bivariate_simple_connectivityMatrix(data)
    % CNN Model - Bivariate Connectivity Matrix

    % Quick test: CNNmodel_Bivariate_simple_connectivityMatrix(EEG.data)
    % Each element (i, j) represents the MSE of the CNN model when using
    % channel i and channel j for prediction.
    
    % Number of channels in the data
    numChannels = size(data, 1); 
    connectivityMatrix = zeros(numChannels, numChannels); % Initialize the connectivity matrix

    % Parallel loop through all channel pairs to calculate MSE
    parfor ch1 = 1:numChannels
        for ch2 = 1:numChannels
            if ch1 ~= ch2
                fprintf('Processing channel pair: (%d, %d)\n', ch1, ch2);
                mseError = CNNmodel_Bivariate_test(ch1, ch2, data);
                connectivityMatrix(ch1, ch2) = mseError; % Store MSE in connectivity matrix
            else
                connectivityMatrix(ch1, ch2) = NaN; % Optional: Set diagonal to NaN
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
    % Extract data for the specified channels
    channel1Data = data(channel1_index, :);
    channel2Data = data(channel2_index, :);

    % Split data into training (80%) and testing (20%) sets
    train_ratio = 0.8;
    train_size = floor(train_ratio * (length(channel1Data) - 1));  

    % Prepare training and testing data
    XTrain = [channel1Data(1, 1:train_size); channel2Data(1, 1:train_size)];  
    YTrain = channel1Data(1, 1:train_size);  
    XTest = [channel1Data(1, train_size + 1:end); channel2Data(1, train_size + 1:end)];
    YTest = channel1Data(1, train_size + 1:end);  

    % Reshape the data for CNN input
    XTrain = reshape(XTrain', [], 2, 1);   
    YTrain = reshape(YTrain, [], 1);       
    XTest = reshape(XTest', [], 2, 1);   
    YTest = reshape(YTest, [], 1);       

    % CNN architecture
    inputSize = 2; 
    numFilters = 32; 
    filterSize = 16; 

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
        'Verbose', false, ...                  
        'ValidationPatience', 5);          % Early stopping patience

    % Train the CNN model
    model = trainnet(XTrain, YTrain, layers, "mse", options);

    % Test the model on the test data
    YPred = predict(model, XTest);

    % Calculate Mean Squared Error
    mseError = mean((YPred - YTest).^2);
end
