function logRatio = ARmodel_FullData(channel1, channel2, data)
    % Computes the log ratio of error variances between two channels and saves models.
    %
    % Parameters:
    %   channel1    : int, index of the first channel
    %   channel2    : int, index of the second channel
    %   data        : matrix, EEG data with channels as rows
    %
    % Returns:
    %   logRatio : float, logarithmic error variance ratio

    % Create directory to save the models just used for testing
    outputDir = 'trained_models_allpairs_AR';
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    order = 16; % Model order

    % Prepare data for univariate and bivariate models
    inputData1 = data(channel1, :)';
    inputData2 = data(channel2, :)';

    Y_target = inputData1(order+1:end); % Target values

    % Create lagged predictors for univariate model on channel1
    X_uni = create_lagged_matrix(inputData1, order);

    % Fit univariate model and compute prediction error
    coefficients_uni = (X_uni' * X_uni) \ (X_uni' * Y_target);
    error_uni = Y_target - X_uni * coefficients_uni;

    % Save univariate model
    save('AR_Univariate_model.mat', 'coefficients_uni');
    % disp(['Univariate model saved']);

    % save(fullfile(outputDir, sprintf('univariate_model_ch%d_ch%d.mat', channel1, channel2)), 'coefficients_uni');


    mse_uni = mean(error_uni.^2);
    % disp(['Mean Squared Error of Univariate Model: ', num2str(mse_uni)]);

    % Create lagged predictors for bivariate model on channels 1 and 2
    X_bi = create_lagged_matrix([inputData1, inputData2], order);

    % Fit bivariate model and compute prediction error
    coefficients_bi = (X_bi' * X_bi) \ (X_bi' * Y_target);
    error_bi = Y_target - X_bi * coefficients_bi;

    % Save bivariate model
    save('AR_Bivariate_model', 'coefficients_bi');
    % disp(['Bivariate model saved']);
    % save(fullfile(outputDir, sprintf('bivariate_model_ch%d_ch%d.mat', channel1, channel2)), 'coefficients_bi');


    % Calculate and display Mean Squared Error for Bivariate model
    mse_bi = mean(error_bi.^2);
    % disp(['Mean Squared Error of Bivariate Model: ', num2str(mse_bi)]);

    % Calculate variances
    variance_uni = var(error_uni);
    variance_bi = var(error_bi);

    % Display variances
    disp(['Variance of Univariate Model Error: ', num2str(variance_uni)]);
    disp(['Variance of Bivariate Model Error: ', num2str(variance_bi)]);

    % Log ratio of variances
    logRatio = log(variance_uni / variance_bi);
    disp(['Log Ratio: ', num2str(logRatio)])

    % % Set the time limit to display only 1 second of data (160 samples)
    % timeLimit = 160;  % For 1 second of data at 160 Hz
    % 
    % % Plot actual vs predicted values for both models in one plot
    % figure;
    % hold on;
    % 
    % % Plot actual values
    % plot(1:timeLimit, Y_target(1:timeLimit), 'b', 'LineWidth', 1.0);
    % 
    % % Plot predicted values from univariate model
    % plot(1:timeLimit, X_uni(1:timeLimit, :) * coefficients_uni, 'r', 'LineWidth', 0.5);
    % 
    % % Plot predicted values from bivariate model
    % plot(1:timeLimit, X_bi(1:timeLimit, :) * coefficients_bi, 'g', 'LineWidth', 0.5);
    % 
    % % Set the plot labels and title
    % title('Actual vs Predicted Values (Univariate & Bivariate Models)');
    % legend('Actual', 'Univariate Predicted', 'Bivariate Predicted');
    % xlabel('Time Step');
    % ylabel('Signal Value');
    % hold off;

end

function laggedMatrix = create_lagged_matrix(data, order)
    % Creates a lagged matrix for autoregressive model input.
    %
    % Parameters:
    %   data  : matrix, column vectors of input data for each channel
    %   order : int, model order (number of lags)
    %
    % Returns:
    %   laggedMatrix : matrix, matrix with lagged values for autoregressive model

    n = size(data, 1);
    numVars = size(data, 2);
    laggedMatrix = zeros(n - order, order * numVars);

    for i = 1:order
        laggedMatrix(:, (i-1)*numVars + (1:numVars)) = data(order+1-i:end-i, :);
    end
end
