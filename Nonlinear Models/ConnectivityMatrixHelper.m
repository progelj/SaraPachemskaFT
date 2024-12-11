% Define the channel names
channels = {'C3','Cz','C4','Fp1','Fp2','F7','F3','Fz','F4','F8', ...
            'T7','T8','P7','P3','Pz','P4','P8','O1','O2'};

% Directory to save the images
outputDir = 'ar_plots_t1_t2'; % Change this to your desired output directory
if ~exist(outputDir, 'dir')
    mkdir(outputDir); % Create directory if it doesn't exist
end

% Loop over all matrices in logRatioMatrices
for datasetIdx = 1:numel(logRatioMatrices)
    % Extract the current connectivity matrix
    connectivityMatrix = logRatioMatrices{datasetIdx};

    % Plot the connectivity matrix
    figure;
    imagesc(connectivityMatrix);
    colorbar;
    xticks(1:numel(channels));
    yticks(1:numel(channels));
    xticklabels(channels);
    yticklabels(channels);
    title(sprintf('AR Connectivity Matrix - T1 and T2'));
    xlabel('Channel');
    ylabel('Channel');

    % Save the plot as a PNG image
    outputFile = fullfile(outputDir, sprintf('AR_Matrix_Dataset_%d.png', datasetIdx));
    saveas(gcf, outputFile); % Save the current figure
    close(gcf); % Close the figure after saving to avoid too many open figures
end

disp('All plots have been saved.');
