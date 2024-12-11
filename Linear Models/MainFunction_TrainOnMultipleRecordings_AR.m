eeglab; % Initialize EEGLAB

% Paths and Setup
dataFolder = 'C:\Users\Acer\Downloads\eeg-motor-movementimagery-dataset-1.0.0\files'; 
locsFile = 'C:\Users\Acer\Downloads\BCI2000.locs'; % Path to channel location file
channels = {'C3','Cz','C4','Fp1','Fp2','F7','F3','Fz','F4','F8', ...
            'T7','T8','P7','P3','Pz','P4','P8','O1','O2'}; % Channels to be used

% Specify the recordings to use for a single subject
subject = 'S001';  % Example of a single subject to train on
recordingsToUse = {'R03', 'R07', 'R11'};  % Only these recordings will be used for training/testing

% Initialize Data Storage
trainData = [];
trainLabels = [];
testData = [];
testLabels = [];

% Load Data for the Training Subject (e.g., 'S001')
disp('Loading training data...');

subjectFolder = fullfile(dataFolder, subject); 

% Loop through the specified recordings
for j = 1:numel(recordingsToUse)
    recordingFile = fullfile(subjectFolder, [subject recordingsToUse{j} '.edf']); % Construct full path to the specific recording
    if exist(recordingFile, 'file')  % Check if the recording exists
        disp(['Loading recording: ', recordingFile]);

        % Load EEG data
        EEG = pop_biosig(recordingFile);
        EEG = pop_chanedit(EEG, 'load', {locsFile, 'filetype', 'autodetect'});
        EEG = pop_reref(EEG, []); % Reference EEG channels
        EEG = pop_select(EEG, 'channel', channels); % Select specified channels

        if isempty(trainData)
            trainData = EEG.data; % First subject's first recording data
        else
            trainData = cat(2, trainData, EEG.data); % Append each new recording's data
        end

        trainLabels = [trainLabels; repmat({subject}, size(EEG.data, 2), 1)]; % Store the subject label for each sample
    end
end

% Set testing data to the same as training data (since it's the same subject)
testData = trainData;
testLabels = trainLabels;

disp(['Shape of trainData: ', num2str(size(trainData))]);
disp(['Shape of testData: ', num2str(size(testData))]);

% CNN Model Training
disp('Training AR model...');
% CNNmodel_ImprovedGeneralization(1, 3, trainData, trainData);

% Training All Pairs - CNN

logRatios = NaN(19, 19); % Pre-fill with NaN to handle diagonal pairs automatically

% Training All Pairs - AR
for ch1 = 1:19
    for ch2 = 1:19
        if ch1 == ch2
            % Skip training for diagonal pairs
            fprintf('Skipping model training for diagonal pair (%d, %d)...\n', ch1, ch2);
            continue; % Skip the rest of the loop for this iteration
        end
        
        fprintf('Training model for channels (%d, %d)...\n', ch1, ch2);
        logRatios(ch1, ch2) = ARmodel_FullData(ch1, ch2, trainData, testData);
    end
end

% modelDir = 'trained_models';
% logRatios = zeros(19, 19);
% 
% for ch1 = 1:19
%     for ch2 = 1:19
%         fprintf('Testing model for channels (%d, %d)...\n', ch1, ch2);
%         logRatios(ch1, ch2) = testModels_AllPairs(ch1, ch2, dataTest, modelDir);
%     end
% end
% 
% Save the test results
% save('test_results.mat', 'logRatios');


disp('Training and evaluation complete!');
