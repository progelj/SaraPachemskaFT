function connectivityMatrix = CNNmodel_Bivariate_initial_connectivityMatrix(data)
    % Connectivity Matrix. Matrix where each element (i, j) represents the MSE
    % of the CNN model when using channel i and channel j for prediction 

    % Quick test: CNNmodel_Bivariate_initial_connectivityMatrix(EEG.data)

    % Input:
    % - data: EEG data 
    % Output:
    % - connectivityMatrix
    
    numChannels = size(data, 1); % Number of channels 
    disp(numChannels);
    connectivityMatrix = zeros(numChannels, numChannels); % Initialize the connectivity matrix

    
    % for ch1 = 1:numChannels
    %     for ch2 = 1:numChannels
    %         if ch1 ~= ch2 % Exclude self-connections
    %             % Display the current channel pair being processed
    %             fprintf('Processing channel pair: (%d, %d)\n', ch1, ch2);
    %             % Train CNN model using ch1 and ch2 and obtain MSE
    %             mseError = CNNmodel_Bivariate_test(ch1, ch2, data);
    % 
    %             % Store the MSE in the connectivity matrix
    %             connectivityMatrix(ch1, ch2) = mseError;
    %         else
    %             connectivityMatrix(ch1, ch2) = NaN; % Optional: Set diagonal to NaN or 0
    %         end
    %     end
    % end

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
        sequenceInputLayer(inputSize)
        convolution1dLayer(filterSize, numFilters, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.2)
        convolution1dLayer(filterSize, numFilters*2, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.2)
        fullyConnectedLayer(100)
        reluLayer
        fullyConnectedLayer(1)
    ];                                             

    options = trainingOptions('adam', ...
        'MaxEpochs', 200, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.001, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {XTest, YTest}, ...
        'ValidationFrequency', 10, ...
        'Verbose', false, ...
        'ValidationPatience', 5);

    % Train the CNN model
    model = trainnet(XTrain, YTrain, layers, "mse", options);

    % Test the model on the test data
    YPred = predict(model, XTest);

    % Calculate Mean Squared Error
    mseError = mean((YPred - YTest).^2);
end
