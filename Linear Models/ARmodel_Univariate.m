function ARmodel_Univariate(channel_index, data, order)
    % Augroregressive Model - Univariate

    % Quick test: ARmodel_Univariate(1, EEG.data, 10)

    % Input:
    % - channel_index: index of the EEG channel (electrode) to use for prediction
    % - data: EEG data (univariate time series)
    % - order: number of lags to use / number of history points
    
    % Extract the channel data 
    inputData = data(channel_index, :);  
    inputData = inputData'; % Transpose to column vector

    % Split data into training (80%) and testing (20%) sets
    train_ratio = 0.8;
    num_samples = length(inputData);  % Number of time points
    train_size = floor(train_ratio * num_samples);

    trainData = inputData(1:train_size);  % Training data
    testData = inputData(train_size+1:end);  % Testing data

    % lags of data
    Xtrain = [];
    Ytrain = trainData(order+1:end);  % Target

    for i = 1:order
        Xtrain = [Xtrain, trainData(order+1-i:end-i)];
    end

    % X_train * coefficients = Y_train
    coefficients = (Xtrain' * Xtrain) \ (Xtrain' * Ytrain);

    % Predictions
    Xtest = [];
    for i = 1:order
        Xtest = [Xtest, testData(order+1-i:end-i)];
    end
    YPred = Xtest * coefficients;  % Predicted values for the test data
    
    % Align the predictions with the test data
    % pad the first 'order' predictions with NaN, so it matches the size of testData
    YPred_full = [nan(order, 1); YPred];

    % Plot
    % for plotting we have the full row with NaN so the signals can be well
    % analyzed
    figure;
    plot(YPred_full, 'r', 'MarkerSize', 1, 'LineWidth', 1, 'DisplayName', 'Predicted'); 
    hold on;
    plot(testData, 'b'); % Actual values in blue
    legend('Predicted', 'Actual');
    title(['AR MODEL - Predicted vs Actual for Channel ' num2str(channel_index)]);

    % Mean Squared Error (MSE)
    % Skip the first 'order' points when comparing predictions and actual data
    % In the YPred we have 'order' less values , so we skip the first in
    % the actual test data
    mseError = mean((YPred - testData(order+1:end)).^2);
    disp(['AR model - Mean Squared Error on Test Data: ', num2str(mseError)]);

    impulse_response_AR(order, coefficients);
end

function impulse_response_AR(order, coefficients)
    % Calculate and plot the impulse response for the AR model
    % Order is the number of lags used
    % Coefficients are the AR model coefficients

    % Generate the impulse signal
    impulse = zeros(1, 100); % Length of the response
    impulse(1) = 1; % Unit impulse
    
    % Calculate the impulse response
    impulse_response = filter(1, [1; -coefficients], impulse);
    
    % Plot the impulse response
    figure;
    plot(impulse_response, 'LineWidth', 2);
    title('Impulse Response of AR Model');
    xlabel('Samples');
    ylabel('Amplitude');
end
