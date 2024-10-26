function ARmodel_Univariate(channel_index, data, max_order)
    
    % find the optimal order based on MSE,
    % impulse response

    % Input:
    % - channel_index: index of the EEG channel (electrode) to use for prediction
    % - data: EEG data (univariate time series)
    % - max_order: maximum AR order to evaluate

    % Extract the channel data 
    inputData = data(channel_index, :); 

    % Transpose to column vector
    inputData = inputData';

    % Split data into training (80%) and testing (20%) sets
    train_ratio = 0.8;
    num_samples = length(inputData);  % Number of time points
    train_size = floor(train_ratio * num_samples);

    % Display lengths of training and testing sets once
    disp(['Training set length: ', num2str(train_size)]);
    disp(['Testing set length: ', num2str(num_samples - train_size)]);

    % Initialize an array to store MSE for each order
    mse_values = zeros(max_order, 1);
    coefficients_store = cell(max_order, 1); % Store coefficients for each order

    % Loop over different model orders to calculate MSE
    for order = 1:max_order
        [mse_values(order), coefficients_store{order}] = run_AR_model(inputData, train_size, order, false);  % Compute MSE for each order
    end

    % Find the order with minimum MSE
    [min_mse, optimal_order] = min(mse_values);
    optimal_coefficients = coefficients_store{optimal_order};

    % Display the optimal order and corresponding MSE
    disp(['Optimal Order (MSE): ', num2str(optimal_order)]);
    disp(['Minimum MSE: ', num2str(min_mse)]);
    
    % Plot MSE vs. Order
    figure;
    plot(1:max_order, mse_values, '-o', 'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    title('MSE vs. Model Order');
    xlabel('Model Order');
    ylabel('Mean Squared Error (MSE)');
    legend('MSE per Order');
    hold off;

    % Run the model one more time with the optimal order for final prediction and plot
    run_AR_model(inputData, train_size, optimal_order, true);

    % Display the impulse response for the optimal order
    impulse_response_AR(optimal_order, optimal_coefficients);
end

function [mseError, coefficients] = run_AR_model(inputData, train_size, order, plot_flag)

    % Define training and testing data
    trainData = inputData(1:train_size);  % Training data
    testData = inputData(train_size+1:end);  % Testing data

    % Prepare lags of data for training
    Xtrain = [];
    Ytrain = trainData(order+1:end);  % Target

    for i = 1:order
        Xtrain = [Xtrain, trainData(order+1-i:end-i)];
    end

    % X_train * coefficients = Y_train
    coefficients = (Xtrain' * Xtrain) \ (Xtrain' * Ytrain);

    % Prepare testing data for predictions
    Xtest = [];
    for i = 1:order
        Xtest = [Xtest, testData(order+1-i:end-i)];
    end
    YPred = Xtest * coefficients;  % Predicted values for the test data
    
    % Align the predictions with the test data for plotting
    YPred_full = [nan(order, 1); YPred];

    % Only plot predictions vs actual if plot_flag is true
    if plot_flag
        figure;
        plot(YPred_full, 'r', 'MarkerSize', 1, 'LineWidth', 1, 'DisplayName', 'Predicted'); 
        hold on;
        plot(testData, 'b'); % Actual values in blue
        legend('Predicted', 'Actual');
        title(['AR MODEL - Predicted vs Actual for Optimal Order ' num2str(order)]);
    end

    % Calculate and return Mean Squared Error (MSE)
    mseError = mean((YPred - testData(order+1:end)).^2);
    disp(['AR model - Mean Squared Error on Test Data (Order ', num2str(order), '): ', num2str(mseError)]);
end

function impulse_response_AR(order, coefficients)
    % Plot the impulse response of the AR model with the given order and coefficients.
    % Input:
    % - order: number of lags used
    % - coefficients: AR model coefficients

    % Generate the impulse signal
    impulse = zeros(1, 100); % Length of the response
    impulse(1) = 1; % Unit impulse
    
    % Calculate the impulse response
    impulse_response = filter(1, [1; -coefficients], impulse);
    
    % Plot the impulse response
    figure;
    plot(impulse_response, 'LineWidth', 2);
    title(['Impulse Response of AR Model (Order ' num2str(order) ')']);
    xlabel('Samples');
    ylabel('Amplitude');
end
