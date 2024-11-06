function connectivityMatrix = ARmodel_Bivariate_connectivityMatrix_FullData(data)
    % Connectivity Matrix - Full Data. Each element (i, j) represents the MSE
    % of the AR model when using channel i and channel j for prediction on the full data.

    % Quick test: ARmodel_Bivariate_connectivityMatrix_FullData(EEG.data)

    % Input:
    % - data: EEG data 
    % Output:
    % - connectivityMatrix
    
    numChannels = size(data, 1); 
    connectivityMatrix = zeros(numChannels, numChannels); % Initialize the connectivity matrix

    % Compute connectivity for each pair of channels
    for ch1 = 1:numChannels
        for ch2 = 1:numChannels
            if ch1 ~= ch2
                fprintf('Processing channel pair: (%d, %d)\n', ch1, ch2);
                mseError = ARmodel_Bivariate_test_FullData(ch1, ch2, data);
                connectivityMatrix(ch1, ch2) = mseError;
            else
                connectivityMatrix(ch1, ch2) = NaN;  % Diagonal elements are NaN
            end
        end
    end

    % Display the connectivity matrix as a heatmap for visualization
    figure;
    imagesc(connectivityMatrix);
    colorbar;
    
    % Adjust the axes to place (1,1) in the bottom-left corner
    set(gca, 'YDir', 'normal');  % Reverses the y-axis direction so 1 is at the bottom
    
    % Set tick labels to match channel numbers starting from 1
    xticks(1:numChannels);
    yticks(1:numChannels);
    
    % Add titles and labels
    title('Connectivity Matrix - MSE between Channel Pairs - Linear model');
    xlabel('Channel');
    ylabel('Channel');
end

function mseError = ARmodel_Bivariate_test_FullData(channel1_index, channel2_index, data)
    % Bivariate AR Model Test with Full Data (no train-test split)

    order = 16;

    % Extract the channels data 
    inputData1 = data(channel1_index, :);  % First channel 
    inputData2 = data(channel2_index, :);  % Second channel 
    
    % Transpose to column vectors
    inputData1 = inputData1';  
    inputData2 = inputData2';

    disp(inputData1);

    % Prepare lagged matrices for both channels on the full dataset
    Xfull = [];
    Yfull = inputData1(order+1:end);  % Target for channel 1 (output)

    for i = 1:order
        % Include lags of both channels
        Xfull = [Xfull, inputData1(order+1-i:end-i), inputData2(order+1-i:end-i)];
    end

    % Fit the model on the full dataset
    coefficients = (Xfull' * Xfull) \ (Xfull' * Yfull);

    % Predict on the full dataset
    YPred_full = Xfull * coefficients;  % Predicted values for the full data
    
    % Mean Squared Error (MSE) on the full data (skipping initial 'order' lags)
    mseError = mean((YPred_full - inputData1(order+1:end)).^2);
end
