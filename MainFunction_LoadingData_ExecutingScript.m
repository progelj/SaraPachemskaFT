% Main script to compute the connectivity matrix using CNN model log error variance ratios

% Paths to the EEG data and channel location files
dataFile = 'C:\Users\Acer\Downloads\S001R01.edf';
locsFile = 'C:\Users\Acer\Downloads\BCI2000.locs';

% Load EEG data
eeglab; % Initialize EEGLAB
EEG = pop_biosig(dataFile);

% Load electrode positions
EEG = pop_chanedit(EEG, 'load', {locsFile, 'filetype', 'autodetect'});

% Re-reference and select 19 channels
EEG = pop_reref(EEG, []);
channels = {'C3','Cz','C4','Fp1','Fp2','F7','F3','Fz','F4','F8',...
            'T7','T8','P7','P3','Pz','P4','P8','O1','O2'};
EEG = pop_select(EEG, 'channel', channels);

% Executing the script
% ConnectivityMatirxAR(EEG.data, channels);
ConnectivityMatrixCNN(EEG.data, channels);
 % CNNmodel_Simple_FullData(1, 2, EEG.data);

