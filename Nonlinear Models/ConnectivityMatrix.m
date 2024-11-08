function connectivityMatrix = ConnectivityMatrix(data)
    % compute_CNN_connectivity_matrix
    % Computes the connectivity matrix using CNN model's logarithmic error variance ratios.
    %
    % Parameters:
    %   data : matrix, EEG data with channels as rows
    %
    % Returns:
    %   connectivityMatrix : matrix, connectivity matrix with log error variance ratios

    numChannels = size(data, 1);
    connectivityMatrix = NaN(numChannels); % Initialize with NaNs

    % Compute connectivity for each pair of channels using CNN model
    for ch1 = 1:numChannels
        for ch2 = 1:numChannels
            if ch1 ~= ch2
                fprintf('Processing channel pair: (%d, %d)\n', ch1, ch2);
                connectivityMatrix(ch1, ch2) = ...
                    CNNmodel_Simple_FullData(ch1, ch2, data);
            end
        end
    end
end


