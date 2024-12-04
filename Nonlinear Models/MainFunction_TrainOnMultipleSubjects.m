eeglab; % Initialize EEGLAB

% Paths and Setup
dataFolder = 'C:\Users\Acer\Downloads\eeg-motor-movementimagery-dataset-1.0.0\files\'; % Path to EEG data folder
locsFile = 'C:\Users\Acer\Downloads\BCI2000.locs'; % Path to channel location file
channels = {'C3','Cz','C4','Fp1','Fp2','F7','F3','Fz','F4','F8',...
            'T7','T8','P7','P3','Pz','P4','P8','O1','O2'};


% Subject List 
subjects = arrayfun(@(x) sprintf('S%03d', x), 1:109, 'UniformOutput', false);

% Split into Training and Testing Sets
numSubjects = numel(subjects);
rng(42); % Set seed for reproducibility
shuffledSubjects = subjects(randperm(numSubjects));
trainSubjects = shuffledSubjects(1:floor(0.8 * numSubjects)); % 80% for training
testSubjects = shuffledSubjects(floor(0.8 * numSubjects) + 1:end); % 20% for testing

% Initialize Training Data Storage
trainData = [];
trainLabels = [];
testData = [];
testLabels = [];
% channel1_train = 1;
% channel2_train = 2;
% 
% channels_train = {'C3', 'C4'}; 
% channels_test = {'Fp1', 'Fp2'};
% 
% 
% channel1_test = 4;
% channel2_test = 5;

% Load Data for Training Subjects
disp('Loading training data...');
for i = 1:numel(trainSubjects)
    subject = trainSubjects{i};
    recordingFile = fullfile(dataFolder, subject, [subject 'R03.edf']); % Using R03 for consistency
    display(recordingFile);
    % Display the length of the EEG recording (number of time points)
    disp(['Length of EEG.data recording: ', num2str(size(EEG.data, 2))]);

    % Load EEG data
    EEG = pop_biosig(recordingFile);
    EEG = pop_chanedit(EEG, 'load', {locsFile, 'filetype', 'autodetect'});
    EEG = pop_reref(EEG, []);
    EEG = pop_select(EEG, 'channel', channels); % Select only two channels
    
    if isempty(trainData)
        trainData = EEG.data; % First subject's data
    else
        trainData = cat(2, trainData, EEG.data); % Append time series
    end

   
    trainLabels = [trainLabels; repmat({subject}, size(EEG.data, 2), 1)];
end

% Load Data for Testing Subjects
disp('Loading testing data...');
for i = 1:numel(testSubjects)
    subject = testSubjects{i};
    recordingFile = fullfile(dataFolder, subject, [subject 'R03.edf']); % Using R03 for consistency
    
    % Load EEG data
    EEG = pop_biosig(recordingFile);
    EEG = pop_chanedit(EEG, 'load', {locsFile, 'filetype', 'autodetect'});
    EEG = pop_reref(EEG, []);
    EEG = pop_select(EEG, 'channel', channels); 
    if isempty(testData)
        testData = EEG.data; % First subject's data
    else
        testData = cat(2, testData, EEG.data); % Append time series
    end

    testLabels = [testLabels; repmat({subject}, size(EEG.data, 2), 1)];
end

disp(['Shape of trainData: ', num2str(size(trainData))]);
disp(['Shape of testData: ', num2str(size(testData))]);
% CNN Model Training
disp('Training CNN model...');

% Training on training data
CNNmodel_ImprovedGeneralization(1, 3, trainData, trainData);
testSavedModels(4, 5, testData);

disp('Training and evaluation complete!');
