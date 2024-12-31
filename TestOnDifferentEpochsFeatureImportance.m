eeglab
% Path to the .edf file
filepath = 'C:\Users\Acer\Downloads\eeg-motor-movementimagery-dataset-1.0.0\files\S001';
filename = 'S001R03.edf';
fullpath = fullfile(filepath, filename);

channels = {'C3','Cz','C4','Fp1','Fp2','F7','F3','Fz','F4','F8',...
            'T7','T8','P7','P3','Pz','P4','P8','O1','O2'};


EEG = pop_biosig(fullpath,'importannot','on');
EEG = pop_chanedit(EEG, 'load',{['C:\Users\Acer\Downloads\BCI2000.locs'] 'filetype' 'autodetect'});

EEG = pop_reref(EEG, []);
EEG = pop_select(EEG, 'channel', channels);

event_file = 'C:\Users\Acer\Desktop\events_S001R03.txt';  % path to the text file with epochs

% Import event information
EEG = pop_importevent(EEG, 'event', event_file, 'fields', {'latency', 'type', 'duration'}, ...
    'skipline', 0, 'timeunit', 1);

% Check EEG structure after importing events
EEG = eeg_checkset(EEG);

EEG_t0 = pop_epoch( EEG, {'T0'}, [0 4.2]); % epoching REST
EEG_t1_t2 = pop_epoch( EEG, {'T1','T2'}, [0 4.1]); % epoching MOTION


% Loop through all epochs
num_epochs = size(EEG_t1_t2.data, 3); % Total number of epochs
for epoch = 1:num_epochs
    % Extract 2D data for the current epoch
    epoch_data = EEG_t1_t2.data(:, :, epoch);

    % Dynamically create variable names like EEG1, EEG2, ..., EEG15
    eval(sprintf('EEG%d = epoch_data;', epoch));
end

% Path to the trained models
% modelPath = 'trained_models_allpairs';
modelPath = 'trained_models_allpairs_AR';

logRatioMatrices = cell(15, 1); % Cell array to store results

for datasetIdx = 1:15
    EEGdata = eval(sprintf('EEG%d', datasetIdx)); % Dynamically load EEG dataset
    logRatioMatrix = NaN(19, 19); % Initialize logRatioMatrix for the current dataset

    % Perform testing for all channel pairs - AR
    for ch1 = 1:19
        for ch2 = 1:19
            if ch1 ~= ch2
                % Load the trained model for the current channel pair
                modelFileUni = fullfile(modelPath, sprintf('univariate_model_ch%d_ch%d.mat', ch1, ch2));
                modelFileBi = fullfile(modelPath, sprintf('bivariate_model_ch%d_ch%d.mat', ch1, ch2));

                if isfile(modelFileUni) && isfile(modelFileBi)
                    uniModelData = load(modelFileUni);
                    biModelData = load(modelFileBi);

                    coefficients_uni = uniModelData.coefficients_uni;
                    coefficients_bi = biModelData.coefficients_bi;

                    order = 16; % Model order used during training

                    % Extract data for specified channels
                    channel1Data = EEGdata(ch1, :)';
                    channel2Data = EEGdata(ch2, :)';

                    % Prepare target data
                    Y_target = channel1Data(order+1:end);

                    % Create lagged predictors for the univariate model
                    X_uni = create_lagged_matrix(channel1Data, order);

                    % Compute predictions for univariate model
                    YPred_uni = X_uni * coefficients_uni;

                    % Compute error for univariate model
                    error_uni = Y_target - YPred_uni;
                    var_uni = var(error_uni);

                    % Create lagged predictors for the bivariate model
                    X_bi = create_lagged_matrix([channel1Data, channel2Data], order);

                    % Compute predictions for bivariate model
                    YPred_bi = X_bi * coefficients_bi;

                    % Compute error for bivariate model
                    error_bi = Y_target - YPred_bi;
                    var_bi = var(error_bi);

                    % Compute and store the log ratio of variances
                    logRatioMatrix(ch1, ch2) = log(var_uni / var_bi);
                else
                    % Handle missing models gracefully
                    fprintf('Model for channels %d and %d not found.\n', ch1, ch2);
                    logRatioMatrix(ch1, ch2) = NaN; % Assign NaN for missing model
                end
            else
                % Diagonal elements (self-pair) can be set to NaN or skipped
                logRatioMatrix(ch1, ch2) = NaN;
            end
        end
    end

    % Store the result in the cell array
     logRatioMatrices{datasetIdx} = logRatioMatrix;
end

function laggedMatrix = create_lagged_matrix(data, order)
    % Creates a lagged matrix for autoregressive model input.
    n = size(data, 1);
    numVars = size(data, 2);
    laggedMatrix = zeros(n - order, order * numVars);

    for i = 1:order
        laggedMatrix(:, (i-1)*numVars + (1:numVars)) = data(order+1-i:end-i, :);
    end
end

% 
% 
% for datasetIdx = 1:15
%     EEGdata = eval(sprintf('EEG%d', datasetIdx));
%     logRatioMatrix = NaN(19, 19);
%     % Perform testing for all channel pairs - CNN
%     for ch1 = 1:19
%         for ch2 = 1:19
%             if ch1 ~= ch2
%                 % Load the trained model for the current channel pair
%                 modelFileUni = fullfile(modelPath, sprintf('univariate_model_ch%d_ch%d.mat', ch1, ch2));
%                 modelFileBi = fullfile(modelPath, sprintf('bivariate_model_ch%d_ch%d.mat', ch1, ch2));
% 
%                 if isfile(modelFileUni) && isfile(modelFileBi)
%                     % Load the univariate and bivariate models
%                     uniModelData = load(modelFileUni);
%                     biModelData = load(modelFileBi);
%                     model_uni = uniModelData.model_uni;
%                     model_bi = biModelData.model_bi;
% 
%                     % Extract the data for the channel pair from EEGdata
%                     channel1Data = EEGdata(ch1, :);
%                     channel2Data = EEGdata(ch2, :);
% 
%                     % Reshape for model input
%                     XVal_uni = reshape(channel1Data, [], 1, 1);
%                     YVal_uni = reshape(channel1Data, [], 1);
% 
%                     XVal_bi = [channel1Data; channel2Data]';
%                     XVal_bi = reshape(XVal_bi, [], 2, 1);
%                     YVal_bi = reshape(channel1Data, [], 1);
% 
%                     % Predict and compute error for univariate model
%                     YPred_uni = predict(model_uni, XVal_uni);
%                     error_uni = YVal_uni - YPred_uni;
%                     var_uni = var(error_uni);
% 
%                     % Predict and compute error for bivariate model
%                     YPred_bi = predict(model_bi, XVal_bi);
%                     error_bi = YVal_bi - YPred_bi;
%                     var_bi = var(error_bi);
% 
%                     % Compute log ratio
%                     logRatioMatrix(ch1, ch2) = log(var_uni / var_bi);
%                 else
%                     % Handle missing models gracefully
%                     fprintf('Model for channels %d and %d not found.\n', ch1, ch2);
%                     logRatioMatrix(ch1, ch2) = NaN; % Assign NaN for missing model
%                 end
%             else
%                 % Diagonal elements (self-pair) can be set to 0 or skipped
%                 logRatioMatrix(ch1, ch2) = NaN; 
%             end
%         end
%     end
%      % Store the result in the cell array
%     logRatioMatrices{datasetIdx} = logRatioMatrix;
% end


finalMatrix_t1_t2 = NaN(15, 361); % Initialize a matrix to hold all rows

for datasetIdx = 1:15
    % Get the log ratio matrix for the current dataset
    logRatioMatrix = logRatioMatrices{datasetIdx};

    % Flatten the 19x19 matrix into a single row of length 361
    flatRow = reshape(logRatioMatrix', 1, []); % Flatten column-major order

    % Store the row in the final matrix
    finalMatrix_t1_t2(datasetIdx, :) = flatRow;
end







