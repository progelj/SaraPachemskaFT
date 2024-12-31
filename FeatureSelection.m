% Script to merge matrices, perform feature importance, and save results

% Load the matrices (assuming they are loaded or already in the workspace)
% finalMatrix_t0 and finalMatrix_t1_t2 should be pre-loaded

% Merge the two matrices
mergedMatrix = [finalMatrix_t0; finalMatrix_t1_t2];

% Create class labels
% Assuming first half corresponds to one class and the second half to another
numSamples_t0 = size(finalMatrix_t0, 1);
numSamples_t1_t2 = size(finalMatrix_t1_t2, 1);
classLabels = [ones(numSamples_t0, 1); 2 * ones(numSamples_t1_t2, 1)];

% Perform feature selection using fscmrmr
% fscmrmr requires features as columns and labels as a vector
[featureIndices, featureScores] = fscmrmr(mergedMatrix, classLabels);

% Extract the top 30 features
topFeatures = featureIndices(1:30);

% Select the top 30 features from the merged matrix
selectedFeatures = mergedMatrix(:, topFeatures);

% Create a structure to save the results
selectedFeaturesStruct = struct();
selectedFeaturesStruct.selectedFeatures = selectedFeatures;
selectedFeaturesStruct.featureIndices = topFeatures;

% Save the results in a .mat file
save('selectedFeaturesData_AR.mat', 'selectedFeaturesStruct');

% Display success message
disp('Feature selection complete. Top 30 features saved in selectedFeaturesData.mat');
