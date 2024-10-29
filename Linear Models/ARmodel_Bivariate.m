function ARmodel_Bivariate(channel_index1, channel_index2, data, order)
    % Augroregressive Model - Bivariate
    % Predict channel 1 based on its own history and the history of channel 2

    % Quick test: ARmodel_Bivariate(1, 2, EEG.data, 10)

    % Input:
    % - channel_index1: index of the EEG channel to predict (output)
    % - channel_index2: index of the EEG channel to use as an input (predictor)
    % - data: EEG data (multivariate time series)
    % - order: number of lags to use / number of history points
    
    % Extract the channels data 
    inputData1 = data(channel_index1, :);  % First channel (output)
    inputData2 = data(channel_index2, :);  % Second channel (predictor)
    
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
    
    % Align the predictions with the test data
    % pad the first 'order' predictions with NaN, so it matches the size of
    % testData1
    YPred_full = [nan(order, 1); YPred];

    % Plot
    % for plotting we have the full row with NaN so the signals can be well
    % analyzed
    figure;
    plot(YPred_full, 'r', 'MarkerSize', 1, 'LineWidth', 1, 'DisplayName', 'Predicted'); 
    hold on;
    plot(testData1, 'b'); 
    legend('Predicted', 'Actual');
    title(['Bivariate AR MODEL - Predicted vs Actual for Channel ' num2str(channel_index1) ' using Channel ' num2str(channel_index2)]);

    % Mean Squared Error (MSE)
    mseError = mean((YPred - testData1(order+1:end)).^2);
    disp(['Bivariate AR model - Mean Squared Error on Test Data: ', num2str(mseError)]);

    impulse_response_AR_bivariate(order, coefficients);
end

function impulse_response_AR_bivariate(order, coefficients)

    % Order is the number of lags used
    % Coefficients are the AR model coefficients (including for both channels)

    % Generate the impulse signal
    impulse = zeros(1, 100); % Length of the response
    impulse(1) = 1; % Unit impulse
    
    % Calculate the impulse response (filter uses both channels' coefficients)
    impulse_response = filter(1, [1; -coefficients], impulse);
    
    % Plot the impulse response
    figure;
    plot(impulse_response, 'LineWidth', 2);
    title('Impulse Response of Bivariate AR Model');
    xlabel('Samples');
    ylabel('Amplitude');
end
