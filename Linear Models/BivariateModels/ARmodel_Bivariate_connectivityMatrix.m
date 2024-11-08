function connectivityMatrix = ARmodel_Bivariate_connectivityMatrix(data)
    % Connectivity Matrix. Matrix where each element (i, j) represents the MSE
    % of the CNN model when using channel i and channel j for prediction 

    % Quick test: ARmodel_Bivariate_connectivityMatrix(EEG.data)

    % Input:
    % - data: EEG data 
    % Output:
    % - connectivityMatrix
    
    numChannels = size(data, 1); % Number of channels 
    disp(numChannels);
    connectivityMatrix = zeros(numChannels, numChannels); % Initialize the connectivity matrix

    % Computing 
    for ch1 = 1:numChannels
        for ch2 = 1:numChannels
            if ch1 ~= ch2
                fprintf('Processing channel pair: (%d, %d)\n', ch1, ch2);
                mseError = ARmodel_Bivariate_test(ch1, ch2, data);
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

function mseError = ARmodel_Bivariate_test(channel1_index, channel2_index, data)

    order = 16;

    % Extract the channels data 
    inputData1 = data(channel1_index, :);  % First channel (output)
    inputData2 = data(channel2_index, :);  % Second channel (predictor)
    
    % Transpose both to column vectors
    inputData1 = inputData1';  
    inputData2 = inputData2';  

    % Split data into training (80%) and testing (20%) sets
    train_ratio = 0.8;
    num_samples = length(inputData1);  % Number of time points
    train_size = floor(train_ratio * num_samples);

    trainData1 = inputData1(1:train_size);  % Training data for channel 1
    trainData2 = inputData2(1:train_size);  % Training data for channel 2

    testData1 = inputData1(train_size+1:end);  % Testing data for channel 1
    testData2 = inputData2(train_size+1:end);  % Testing data for channel 2

    % Lagged matrices for both channels
    Xtrain = [];
    Ytrain = trainData1(order+1:end);  % Target for channel 1 (output)

    for i = 1:order
        % Include lags of both channel 1 and channel 2
        Xtrain = [Xtrain, trainData1(order+1-i:end-i), trainData2(order+1-i:end-i)];
    end

    coefficients = (Xtrain' * Xtrain) \ (Xtrain' * Ytrain);

    % Predictions for the test set
    Xtest = [];
    for i = 1:order
        % Include lags of both channel 1 and channel 2 in the test set
        Xtest = [Xtest, testData1(order+1-i:end-i), testData2(order+1-i:end-i)];
    end
    YPred = Xtest * coefficients;  % Predicted values for the test data
    
    % Mean Squared Error (MSE)
    mseError = mean((YPred - testData1(order+1:end)).^2);
    
end
