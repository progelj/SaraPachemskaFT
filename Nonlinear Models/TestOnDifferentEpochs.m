eeglab
% Path to the .edf file
filepath = 'C:\Users\Acer\Downloads\eeg-motor-movementimagery-dataset-1.0.0\files\S001';
filename = 'S001R07.edf';
fullpath = fullfile(filepath, filename);

channels = {'C3','Cz','C4','Fp1','Fp2','F7','F3','Fz','F4','F8',...
            'T7','T8','P7','P3','Pz','P4','P8','O1','O2'};


EEG = pop_biosig(fullpath,'importannot','on');
EEG = pop_chanedit(EEG, 'load',{['C:\Users\Acer\Downloads\BCI2000.locs'] 'filetype' 'autodetect'});

EEG = pop_reref(EEG, []);
EEG = pop_select(EEG, 'channel', channels);

event_file = 'C:\Users\Acer\Desktop\events_S001R03.txt';  % Replace with the correct path to your .txt file

% Import event information
EEG = pop_importevent(EEG, 'event', event_file, 'fields', {'latency', 'type', 'duration'}, ...
    'skipline', 0, 'timeunit', 1);

% Check EEG structure after importing events
EEG = eeg_checkset(EEG);

EEG = pop_epoch( EEG, {'T1','T2'}, [0 4.1]); % epoching


% Dimensions: 19 (channels) x 656 (timepoints) x 15 (epochs)

% Loop through all epochs
num_epochs = size(EEG.data, 3); % Total number of epochs
for epoch = 1:num_epochs
    % Extract 2D data for the current epoch
    epoch_data = EEG.data(:, :, epoch);

    % Dynamically create variable names like EEG1, EEG2, ..., EEG15
    eval(sprintf('EEG%d = epoch_data;', epoch));
end

% Initialize a 19x19 matrix to store logRatios
logRatioMatrix = zeros(19, 19);

% Path to the trained models
modelPath = 'trained_models_allpairs';

EEGdata = EEG15;

% Perform testing for all channel pairs
for ch1 = 1:19
    for ch2 = 1:19
        if ch1 ~= ch2
            % Load the trained model for the current channel pair
            modelFileUni = fullfile(modelPath, sprintf('univariate_model_ch%d_ch%d.mat', ch1, ch2));
            modelFileBi = fullfile(modelPath, sprintf('bivariate_model_ch%d_ch%d.mat', ch1, ch2));

            if isfile(modelFileUni) && isfile(modelFileBi)
                % Load the univariate and bivariate models
                uniModelData = load(modelFileUni);
                biModelData = load(modelFileBi);
                model_uni = uniModelData.model_uni;
                model_bi = biModelData.model_bi;

                % Extract the data for the channel pair from EEGdata
                channel1Data = EEGdata(ch1, :);
                channel2Data = EEGdata(ch2, :);

                % Reshape for model input
                XVal_uni = reshape(channel1Data, [], 1, 1);
                YVal_uni = reshape(channel1Data, [], 1);

                XVal_bi = [channel1Data; channel2Data]';
                XVal_bi = reshape(XVal_bi, [], 2, 1);
                YVal_bi = reshape(channel1Data, [], 1);

                % Predict and compute error for univariate model
                YPred_uni = predict(model_uni, XVal_uni);
                error_uni = YVal_uni - YPred_uni;
                var_uni = var(error_uni);

                % Predict and compute error for bivariate model
                YPred_bi = predict(model_bi, XVal_bi);
                error_bi = YVal_bi - YPred_bi;
                var_bi = var(error_bi);

                % Compute log ratio
                logRatioMatrix(ch1, ch2) = log(var_uni / var_bi);
            else
                % Handle missing models gracefully
                fprintf('Model for channels %d and %d not found.\n', ch1, ch2);
                logRatioMatrix(ch1, ch2) = NaN; % Assign NaN for missing model
            end
        else
            % Diagonal elements (self-pair) can be set to 0 or skipped
            logRatioMatrix(ch1, ch2) = NaN; 
        end
    end
end







