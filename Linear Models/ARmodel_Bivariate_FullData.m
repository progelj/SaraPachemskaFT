function ARmodel_Bivariate_FullData(channel_index1, channel_index2, data, order)
    % Bivariate Autoregressive Model using Full Data (no train-test split)
    % Predict channel 1 based on its own history and the history of channel 2

    % Quick test: ARmodel_Bivariate_FullData(1, 1, EEG.data, 16)

    % Input:s
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

    % Prepare lagged matrices for the full dataset
    Xfull = [];
    Yfull = inputData1(order+1:end);  % Target values for channel 1 (output)

    for i = 1:order
        % Include lags of both channel 1 and channel 2
        Xfull = [Xfull, inputData1(order+1-i:end-i), inputData2(order+1-i:end-i)];
    end

    % Fit the model on the full dataset
    coefficients = (Xfull' * Xfull) \ (Xfull' * Yfull);

    % Predictions for the full dataset
    YPred_full = Xfull * coefficients;  % Predicted values for the full data
    
    % Pad the first 'order' points with NaN for alignment
    YPred_full = [nan(order, 1); YPred_full];

    % Plot
    figure;
    plot(YPred_full, 'r', 'MarkerSize', 1, 'LineWidth', 1, 'DisplayName', 'Predicted'); 
    hold on;
    plot(inputData1, 'b');  % Actual values in blue
    legend('Predicted', 'Actual');
    title(['Bivariate AR MODEL - Predicted vs Actual for Channel ' num2str(channel_index1) ' using Channel ' num2str(channel_index2)]);

    % Mean Squared Error (MSE) on the full data (skipping initial lags)
    mseError = mean((YPred_full(order+1:end) - inputData1(order+1:end)).^2, 'omitnan');
    disp(['Bivariate AR model - Mean Squared Error on Full Data: ', num2str(mseError)]);

    % Calculate and plot the impulse response
    impulse_response_AR_bivariate(order, coefficients);
end

function impulse_response_AR_bivariate(order, coefficients)
    % Generate the impulse response for the bivariate AR model

    % Generate the impulse signal
    impulse = zeros(1, 100); % Length of the response
    impulse(1) = 1; % Unit impulse
    
    % Calculate the impulse response using both channels' coefficients
    impulse_response = filter(1, [1; -coefficients], impulse);
    
    % Plot the impulse response
    figure;
    plot(impulse_response, 'LineWidth', 2);
    title('Impulse Response of Bivariate AR Model');
    xlabel('Samples');
    ylabel('Amplitude');
end
