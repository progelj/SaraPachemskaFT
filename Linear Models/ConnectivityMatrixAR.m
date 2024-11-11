function connectivityMatrix = ConnectivityMatrixAR(data, channels)

    % compute_connectivity_matrix
    % Computes the connectivity matrix using logarithmic error variance ratios.
    %
    % Parameters:
    %   data : matrix, EEG data with channels as rows
    %
    % Returns:
    %   connectivityMatrix : matrix, connectivity matrix with log error variance ratios

    numChannels = size(data, 1);
    connectivityMatrix = NaN(numChannels); % Initialize with NaNs

    % Compute connectivity for each pair of channels
    for ch1 = 1:numChannels
        for ch2 = 1:numChannels
            if ch1 ~= ch2
                fprintf('Processing channel pair: (%d, %d)\n', ch1, ch2);
                connectivityMatrix(ch1, ch2) = ...
                    ARmodel_FullData(ch1, ch2, data);
            end
        end
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
    title('AR Connectivity Matrix - Log Ratio of Error Variances');
    xlabel('Channel');
    ylabel('Channel');
end