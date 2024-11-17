function connectivityMatrix = ConnectivityMatrixCNN(data, channels)
    % Compute Connectivity Matrix with Parallel Processing
    % Computes the connectivity matrix using CNN model's logarithmic error variance ratios.
    %
    % Parameters:
    %   data : matrix, EEG data with channels as rows
    %
    % Returns:
    %   connectivityMatrix : matrix, connectivity matrix with log error variance ratios

    numChannels = size(data, 1);
    connectivityMatrix = NaN(numChannels); % Initialize with NaNs

    % parallel processing to compute connectivity for each pair of channels
    parfor ch1 = 1:numChannels
        tempRow = NaN(1, numChannels);  % Temporary row to store results for channel `ch1`
        for ch2 = 1:numChannels
            if ch1 ~= ch2
                fprintf('Processing channel pair: (%d, %d)\n', ch1, ch2);
                % Change the model script in order to test
                tempRow(ch2) = CNNmodel_SimpleOptimizedTest_FullData(ch1, ch2, data);
            end
        end
        connectivityMatrix(ch1, :) = tempRow;  % Update the row in the connectivity matrix
    end

    plotCM(connectivityMatrix, channels);
end

function plotCM(connectivityMatrix, channels)
    figure;
    imagesc(connectivityMatrix);
    colorbar;
    xticks(1:numel(channels));
    yticks(1:numel(channels));
    xticklabels(channels);
    yticklabels(channels);
    title('CNN Connectivity Matrix - Log Ratio of Error Variances');
    xlabel('Channel');
    ylabel('Channel');
end
