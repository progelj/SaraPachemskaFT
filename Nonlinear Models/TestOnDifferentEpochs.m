eeglab
% Path to the .edf file
filepath = 'C:\Users\Acer\Downloads\eeg-motor-movementimagery-dataset-1.0.0\files\S001';
filename = 'S001R03.edf';
fullpath = fullfile(filepath, filename);

EEG = pop_biosig(fullpath,'importannot','off');
EEG = pop_chanedit(EEG, 'load',{['C:\Users\Acer\Downloads\BCI2000.locs'] 'filetype' 'autodetect'});


% Load the annotations
[data, annotations] = edfread(fullpath);
% Create EEGLAB-compatible event structure
% Assuming annotations is a timetable with 'Onset', 'Annotations', and 'Duration' columns



% Rename columns (fields) to match EEGLAB's 'event' structure
% annotations.latency = annotations.Onset; % Rename 'Onset' to 'latency'
% annotations.type = annotations.Annotations; % Rename 'Annotations' to 'type'
% annotations.duration = annotations.Duration; % Rename 'Duration' to 'duration'
% 
% annotations(:, [1, 2]) = [];  % Remove the first and second columns (Onset, Annotations)

% Convert the annotations structure to EEGLAB-compatible event structure
EEG.event = (annotations);

plot(EEG.data);

% EEG = pop_epoch( EEG, {'T1','T2'}, [0 4.1]); % epoching

