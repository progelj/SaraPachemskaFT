function [YPred_full] = ARmodel_Univariate_FullData(channel_index, data)
    % Augroregressive Model - Univariate (no train-test split)

    % Quick test:  ARmodel_Univariate_FullData(1, EEG.data)

    % Input:
    % - channel_index: index of the EEG channel (electrode) to use for prediction
    % - data: EEG data (univariate time series)

    % Extract the channel data 
    inputData = data(channel_index, :); 

    % Transpose to column vector
    inputData = inputData'; 

    % Optimal order for the data
    order = 16;

    % Entire data for fitting
    num_samples = length(inputData);

    % Prepare lagged data for the full dataset
    Xfull = [];
    Yfull = inputData(order+1:end);  % Target values for the full dataset

    for i = 1:order
        Xfull = [Xfull, inputData(order+1-i:end-i)];
    end

    disp(length(Xfull))

    % Fit the model to the full data
    coefficients = (Xfull' * Xfull) \ (Xfull' * Yfull);

    % Predictions on the full dataset
    YPred_full = Xfull * coefficients;  % Predicted values for the full data
    
    % Pad the initial 'order' points with NaN for alignment
    YPred_full = [nan(order, 1); YPred_full];

    % Plot
    figure;
    plot(YPred_full, 'r', 'MarkerSize', 1, 'LineWidth', 1, 'DisplayName', 'Predicted'); 
    hold on;
    plot(inputData, 'b'); % Actual values in blue
    legend('Predicted', 'Actual');
    title(['AR MODEL - Predicted vs Actual for Channel ' num2str(channel_index)]);

    % Calculate and display the Mean Squared Error (MSE) for the full data (skipping NaNs)
    mseError = mean((YPred_full(order+1:end) - inputData(order+1:end)).^2, 'omitnan');
    disp(['AR model - Mean Squared Error on Full Data: ', num2str(mseError)]);

    % Impulse response
    impulse_response_AR(order, coefficients);
end

function impulse_response_AR(order, coefficients)
    % Generate the impulse response for the fitted AR model

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
