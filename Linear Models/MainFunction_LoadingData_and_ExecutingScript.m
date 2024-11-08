clear all;
close all;
eeglab;
% load the data
% EEG = pop_biosig('/home/peter/Projects/EEG-data/eeg-motor-movementimagery-dataset-1.0.0/S001/S001R03.edf');
EEG = pop_biosig('C:\Users\Acer\Downloads\S001R02.edf');

% loading electrode positions:
EEG=pop_chanedit(EEG, 'load',{['C:\Users\Acer\Downloads\BCI2000.locs'] 'filetype' 'autodetect'});

% reduce channel nr. 
% Re-reference to average 
EEG = pop_reref( EEG, []);
% select only 19 out of 64 channels
EEG = pop_select( EEG, 'channel',{'C3','Cz','C4','Fp1','Fp2','F7','F3','Fz','F4','F8','T7','T8','P7','P3','Pz','P4','P8','O1','O2'});

fs=EEG.srate;
nrEl = EEG.nbchan;

%% granger causality computation for all electrode pairs
order = 16;
conG2=zeros(nrEl);
tic
for c1=1:nrEl
    for c2= c1+1:nrEl
        % GC = GCmodel(EEG.data([c1 c2],:), order);
        % %GC = max(GC,[0 0]);
        % conG2(c1,c2) = GC(1);
        % conG2(c2,c1) = GC(2);
    end
end


figure(1);
imagesc(conG2); colorbar; title(['Granger causality, order=' num2str(order) ', ' EEG.setname ]);
xticks([1:nrEl]);xticklabels({EEG.chanlocs.labels}); xtickangle(90)
yticks([1:nrEl]);yticklabels({EEG.chanlocs.labels});
xlabel('influencing electrode');
ylabel('influenced electrode');
axis equal;axis tight;

 %eeglab redraw; % refresh GUI using data defined in the command mode
