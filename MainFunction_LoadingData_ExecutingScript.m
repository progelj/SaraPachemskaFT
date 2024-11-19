% Main script to compute the connectivity matrix using CNN model log error variance ratios

% Paths to the EEG data and channel location files
dataFile = 'C:\Users\Acer\Downloads\eeg-motor-movementimagery-dataset-1.0.0\files\S001\S001R03.edf';

testFile = 'C:\Users\Acer\Downloads\eeg-motor-movementimagery-dataset-1.0.0\files\S001\S001R11.edf';
locsFile = 'C:\Users\Acer\Downloads\BCI2000.locs';

% Load EEG data
eeglab; % Initialize EEGLAB

channels = {'C3','Cz','C4','Fp1','Fp2','F7','F3','Fz','F4','F8',...
            'T7','T8','P7','P3','Pz','P4','P8','O1','O2'};

EEG = pop_biosig(dataFile);
EEG = pop_chanedit(EEG, 'load', {locsFile, 'filetype', 'autodetect'});
EEG = pop_reref(EEG, []);
EEG = pop_select(EEG, 'channel', channels);

% Load Testing EEG data (testFile)
EEG_test = pop_biosig(testFile);
EEG_test = pop_chanedit(EEG_test, 'load', {locsFile, 'filetype', 'autodetect'});
EEG_test = pop_reref(EEG_test, []);
EEG_test = pop_select(EEG_test, 'channel', channels);

% --- Executing the scripts ---1
 

% This is the script for linear model - Computing the Connectivity Matrix
% ConnectivityMatrixAR(EEG.data, channels);

% This is the script for nonlinear - Computing the Connectivity Matrix
% ConnectivityMatrixCNN(EEG.data, channels);

% This is the script if you have already saved model, and you want to test
% on another - Connectivity Matrix
% ConnectivityMatrixCNN(EEG_test.data, channels); 

% This is executing the simple model, with 2 channels
% CNNmodel_SimpleOptimized_FullData(1, 3, EEG.data);

% Loading and Test the CNNmodel
LoadAndTestModel(1,3, EEG_test.data);


