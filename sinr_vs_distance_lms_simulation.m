% clear all; clc; close all;
% 
% % Number of iterations for averaging
% numIterations = 1;
% 
% % Setup for Case 1: TX and RX in opposite lanes
% [SINR_dB_case1, BER_case1] = run_simulation(10, 'Case 1: TX and RX in opposite lanes with interferers across both lanes', numIterations);
% 
% % Setup for Case 2: TX and RX in the same lane
% [SINR_dB_case2, BER_case2] = run_simulation(0, 'Case 2: TX and RX in the same lane with interferers in the opposite lane', numIterations);
% 
% function [avgSINR_dB, avgBER] = run_simulation(lane_offset_RX, title_desc, numIterations)
%     % System Parameters
%     txPower_dBm = 24;
%     carrierFreq = 5.9e9;
%     channelBandwidth = 10e6;
%     noiseFigures_dB = 9:3:24;
%     lambda = 3e8 / carrierFreq;
%     numInterferers = 10;
%     tx_location = [0, 0];
%     lane_width = 18;  % Standard lane width in meters
%     numElementsTX = 8;
%     numElementsRX = 4;
%     numElementsInterferer = 16;
%     elementSpacing = lambda/2;
%     shadowingStdDev_dB = 4;
% 
%     % Mobility Parameters
%     speedTX = 31.5*2.23694;  % Speed of TX vehicle in m/s
%     speedRX = 31.5*2.23694;  % Speed of RX vehicle in m/s
% 
%     % Convert txPower from dBm to watts
%     txPower_W = 10^((txPower_dBm - 30)/10);
% 
%     % Create phased array ULA system objects for TX, RX, Interferers
%     txArray = phased.ULA('NumElements', numElementsTX, 'ElementSpacing', elementSpacing);
%     rxArray = phased.ULA('NumElements', numElementsRX, 'ElementSpacing', elementSpacing);
%     interfererArray = phased.ULA('NumElements', numElementsInterferer, 'ElementSpacing', elementSpacing);
% 
%     % Distance range from TX to RX
%     distances_m = 10:10:400;
% 
%     % Adaptive Beamforming Assumed Gains
%     interferenceReduction_dB = 10;
% 
%     % Initialize array weights for LMS
%     arrayWeightsTX = ones(numElementsTX, 1) / sqrt(numElementsTX);
%     arrayWeightsRX = ones(numElementsRX, 1) / sqrt(numElementsRX);
%     arrayWeightsInterferer = repmat(ones(numElementsInterferer, 1) / sqrt(numElementsInterferer), 1, numInterferers);
% 
%     % Preallocate array for SINR and BER values for each iteration
%     SINR_dB_iterations = zeros(length(distances_m), length(noiseFigures_dB), numIterations);
%     BER_iterations = zeros(length(distances_m), length(noiseFigures_dB), numIterations);
% 
%     for iter = 1:numIterations
%         % Assume constant interference power (in dBm) for simplicity
%         interferencePower_dBm = -20 + 10 * randn(1, numInterferers);
% 
%         % Define locations for interferers randomly across both lanes
%         if lane_offset_RX == 0  % Case 2, interferers only in opposite lane
%             interferer_locations = [rand(numInterferers, 1) * max(distances_m), repmat(lane_width, numInterferers, 1)];
%         else  % Case 1, interferers across both lanes
%             interferer_locations = [rand(numInterferers, 1) * max(distances_m), lane_width * round(rand(numInterferers, 1))];
%         end
% 
%         % Calculate SINR and BER
%         for j = 1:length(noiseFigures_dB)
%             noiseFigure_dB = noiseFigures_dB(j);
%             noisePower_W = 10^((-174 + 10*log10(channelBandwidth) + noiseFigure_dB - 30)/10);
% 
%             for i = 1:length(distances_m)
%                 distance = distances_m(i);
%                 pathLoss_dB = 20*log10(4*pi*distance/lambda);
%                 shadowing_dB = shadowingStdDev_dB * randn;
%                 totalPathLoss_dB = pathLoss_dB + shadowing_dB;
%                 totalReceivedPower_W = 0;
%                 totalInterferencePower_W = 0;
% 
%                 % Calculate interference power
%                 for k = 1:numInterferers
%                     if numElementsRX > numElementsInterferer
%                         weightsRX = arrayWeightsRX(1:numElementsInterferer);
%                     else
%                         weightsRX = [arrayWeightsRX; zeros(numElementsInterferer - numElementsRX, 1)];
%                     end
% 
%                     effectiveInterferencePower_W = 10^((interferencePower_dBm(k) - interferenceReduction_dB - 30)/10) * abs(dot(arrayWeightsInterferer(:,k), weightsRX))^2;
%                     totalInterferencePower_W = totalInterferencePower_W + effectiveInterferencePower_W;
%                 end
% 
%                 % Calculate received power for each cluster and ray
%                 for n = 1:3 % Number of clusters
%                     clusterGain_dB = [-1, -3, -5];
%                     for m = 1:3 % Number of rays per cluster
%                         rayGain_dB = clusterGain_dB(m) - 5 * rand(1);
%                         rayPhase = exp(1i * (10 * randn(1)));
%                         fading = gamrnd(4, 1/4);
% 
%                         rayPathLoss_dB = totalPathLoss_dB - rayGain_dB;
%                         rayPathLoss = 10^(-rayPathLoss_dB / 10);
% 
%                         if numElementsTX > numElementsRX
%                             weightsTX = arrayWeightsTX(1:numElementsRX);
%                         else
%                             weightsTX = [arrayWeightsTX; zeros(numElementsRX - numElementsTX, 1)];
%                         end
%                         receivedPower_W = txPower_W * abs(dot(weightsTX, arrayWeightsRX))^2 * rayPathLoss * fading * abs(rayPhase)^2;
%                         totalReceivedPower_W = totalReceivedPower_W + receivedPower_W;
%                     end
%                 end
% 
%                 % Calculate SINR and BER
%                 SINR = 10 * log10(totalReceivedPower_W / (totalInterferencePower_W + noisePower_W));
%                 SINR_dB_iterations(i, j, iter) = SINR;
% 
%                 SINR_linear = 10^(SINR/10);
%                 BER_iterations(i, j, iter) = qfunc(sqrt(2 * SINR_linear));
%             end
%         end
%     end
% 
%     % Average SINR and BER over all iterations
%     avgSINR_dB = mean(SINR_dB_iterations, 3);
%     avgBER = mean(BER_iterations, 3);
% 
%     % Plot SINR vs. Distance
%     figure;
%     hold on;
%     for j = 1:length(noiseFigures_dB)
%         x = distances_m;
%         y = avgSINR_dB(:, j);
%         pp = spline(x, y);
%         xx = linspace(min(x), max(x), 1000);
%         yy = ppval(pp, xx);
%         plot(xx, yy, 'LineWidth', 2);
%     end
%     grid on;
%     xlabel('Distance between TX and RX (m)');
%     ylabel('SINR (dB)');
%     title(['SINR vs. Distance with Spline Fitting for ' title_desc]);
%     legend(arrayfun(@(x) sprintf('NF = %d dB', x), noiseFigures_dB, 'UniformOutput', false));
%     ylim([-60, -20]);
%     xlim([10,400]);
%     hold off;
% 
%     % Plot BER vs. Distance
%     figure;
%     hold on;
%     for j = 1:length(noiseFigures_dB)
%         x = distances_m;
%         y = avgBER(:, j);
%         pp = spline(x, y);
%         xx = linspace(min(x), max(x), 1000);
%         yy = ppval(pp, xx);
%         plot(xx, yy, 'LineWidth', 2);
%     end
%     grid on;
%     xlabel('Distance between TX and RX (m)');
%     ylabel('BER');
%     title(['BER vs. Distance with Spline Fitting for ' title_desc]);
%     legend(arrayfun(@(x) sprintf('NF = %d dB', x), noiseFigures_dB, 'UniformOutput', false));
%     ylim([0.46, 0.5]);
%     xlim([10,400]);
%     set(gca, 'YScale', 'log');
%     hold off;
% 
%     % Visualization setup for TX, RX, and interferers
%     figure;
%     hold on;
%     grid on;
%     plot(tx_location(1), tx_location(2), 'ko', 'MarkerFaceColor', 'k');
%     plot(interferer_locations(:,1), interferer_locations(:,2), 'r^', 'MarkerFaceColor', 'r');
% 
%     for distance = distances_m
%         rx_location = [distance, lane_offset_RX];
%         plot(rx_location(1), rx_location(2), 'b*');
%     end
% 
%     xlabel('Distance (m)');
%     ylabel('Lane Position (m)');
%     title(title_desc);
%     legend('TX', 'Interferers', 'RX Locations');
%     axis equal;
%     xlim([-20, max(distances_m) + 20]);
%     ylim([-50, 50]);
%  % 3D Visualization
%     figure;
%     surf(noiseFigures_dB, distances_m, avgSINR_dB);
%     xlabel('Noise Figure (dB)');
%     ylabel('Distance (m)');
%     zlabel('SINR (dB)');
%     title(['SINR vs. Distance and Noise Figure with Adaptive Beamforming and Clustered Multipath for ' title_desc]);
%     colorbar;
% 
%     hold off;
% 
%     % Annotations for better understanding
%     annotation('textbox', [.15 .6 .3 .3], 'String', sprintf('Tx Power: %d dBm\nCarrier Freq: %.1f GHz\nNoise Figure: Varies 9-24 dB\nTX ULA Elements: %d\nRX ULA Elements: %d\nInterferer ULA Elements: %d\nIntf. Reduction: %d dB\nNakagami-m Factor: %d', txPower_dBm, carrierFreq/1e9, numElementsTX, numElementsRX, numElementsInterferer, interferenceReduction_dB, 4), 'EdgeColor', 'black', 'BackgroundColor', 'white');
% end
% 
% 

clear all; clc; close all;

% Parameters
numElements = 10; % Number of antenna elements
numInterferers = numElements - 1; % Number of interferers (nulls)

% Number of iterations for averaging
numIterations = 1;

% Initial positions for TX and RX for both cases
initial_tx_location_case1 = [0, 0];
initial_rx_location_case1 = [0, 10]; % RX starts 10 meters away from TX in the same lane

initial_tx_location_case2 = [0, 0];
initial_rx_location_case2 = [18, 10]; % RX starts 10 meters away from TX in the opposite lane with 18m width

% Distance range from TX to RX
distances_m = 10:10:400;

% Positional error standard deviations
positional_errors_std = 1:7; % Standard deviation of positional errors (in meters)

% Pre-generate all possible interferer positions
interferer_positions = generate_all_interferer_positions(numInterferers, 0, 400, 10, [0, 18]);

% SINR and BER results for different numbers of nulls
sinr_results_case1 = zeros(numInterferers, length(distances_m), length(positional_errors_std)); % Adjust size based on numInterferers
ber_results_case1 = zeros(numInterferers, length(distances_m), length(positional_errors_std));
sinr_std_case1 = zeros(numInterferers, length(distances_m), length(positional_errors_std));
ber_std_case1 = zeros(numInterferers, length(distances_m), length(positional_errors_std));

sinr_results_case2 = zeros(numInterferers, length(distances_m), length(positional_errors_std));
ber_results_case2 = zeros(numInterferers, length(distances_m), length(positional_errors_std));
sinr_std_case2 = zeros(numInterferers, length(distances_m), length(positional_errors_std));
ber_std_case2 = zeros(numInterferers, length(distances_m), length(positional_errors_std));

for numNulls = numInterferers:-1:1
    for p = 1:length(positional_errors_std)
        positional_error_std = positional_errors_std(p);

        % Run simulation for Case 1
        [sinr_dB, ber] = run_simulation_with_nulls(numNulls, numInterferers, numIterations, initial_tx_location_case1, initial_rx_location_case1, interferer_positions, distances_m, positional_error_std);
        sinr_results_case1(numInterferers + 1 - numNulls, :, p) = mean(real(sinr_dB), 2); % Averaging SINR over all interferer positions
        ber_results_case1(numInterferers + 1 - numNulls, :, p) = mean(ber, 2); % Averaging BER over all interferer positions
        sinr_std_case1(numInterferers + 1 - numNulls, :, p) = std(real(sinr_dB), 0, 2);
        ber_std_case1(numInterferers + 1 - numNulls, :, p) = std(ber, 0, 2);

        % Run simulation for Case 2
        [sinr_dB, ber] = run_simulation_with_nulls(numNulls, numInterferers, numIterations, initial_tx_location_case2, initial_rx_location_case2, interferer_positions, distances_m, positional_error_std);
        sinr_results_case2(numInterferers + 1 - numNulls, :, p) = mean(real(sinr_dB), 2); % Averaging SINR over all interferer positions
        ber_results_case2(numInterferers + 1 - numNulls, :, p) = mean(ber, 2); % Averaging BER over all interferer positions
        sinr_std_case2(numInterferers + 1 - numNulls, :, p) = std(real(sinr_dB), 0, 2);
        ber_std_case2(numInterferers + 1 - numNulls, :, p) = std(ber, 0, 2);
    end
end

% Define colors for different nulls
colors = lines(numInterferers);

% Plot SINR results for Case 1
figure;
hold on;
for i = 1:numInterferers
    x = distances_m;
    y = squeeze(sinr_results_case1(i, :, 1)); % Plot for first positional error std deviation
    std_dev = squeeze(sinr_std_case1(i, :, 1));

    % Plot error bars for standard deviation
    errorbar(x, y, std_dev, 'o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), 'MarkerEdgeColor', colors(i, :), 'LineWidth', 1, 'CapSize', 5, 'DisplayName', sprintf('%d Nulls Raw Data', numInterferers + 1 - i));

    % Plot mean SINR
    p = polyfit(x, y, 3);
    xx = linspace(min(x), max(x), 1000);
    yy = polyval(p, xx);
    plot(xx, yy, 'LineWidth', 2, 'Color', colors(i, :), 'DisplayName', sprintf('%d Nulls Fit', numInterferers + 1 - i));
end
grid on;
xlabel('Distance between TX and RX (m)');
ylabel('SINR (dB)');
title('SINR vs. Distance with Different Numbers of Nulls Towards Interferers (Case 1)');
legend('show');
hold off;

% Plot SINR results for Case 2
figure;
hold on;
for i = 1:numInterferers
    x = distances_m;
    y = squeeze(sinr_results_case2(i, :, 1)); % Plot for first positional error std deviation
    std_dev = squeeze(sinr_std_case2(i, :, 1));

    % Plot error bars for standard deviation
    errorbar(x, y, std_dev, 'o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), 'MarkerEdgeColor', colors(i, :), 'LineWidth', 1, 'CapSize', 5, 'DisplayName', sprintf('%d Nulls Raw Data', numInterferers + 1 - i));

    % Plot mean SINR
    p = polyfit(x, y, 3);
    xx = linspace(min(x), max(x), 1000);
    yy = polyval(p, xx);
    plot(xx, yy, 'LineWidth', 2, 'Color', colors(i, :), 'DisplayName', sprintf('%d Nulls Fit', numInterferers + 1 - i));
end
grid on;
xlabel('Distance between TX and RX (m)');
ylabel('SINR (dB)');
title('SINR vs. Distance with Different Numbers of Nulls Towards Interferers (Case 2)');
legend('show');
hold off;

% Plot BER results for Case 1
figure;
hold on;
for i = 1:numInterferers
    x = distances_m;
    y = squeeze(ber_results_case1(i, :, 1)); % Plot for first positional error std deviation
    std_dev = squeeze(ber_std_case1(i, :, 1));

    % Plot error bars for standard deviation
    errorbar(x, y, std_dev, 'o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), 'MarkerEdgeColor', colors(i, :), 'LineWidth', 1, 'CapSize', 5, 'DisplayName', sprintf('%d Nulls Raw Data', numInterferers + 1 - i));

    % Plot mean BER
    p = polyfit(x, y, 3);
    xx = linspace(min(x), max(x), 1000);
    yy = polyval(p, xx);
    plot(xx, yy, 'LineWidth', 2, 'Color', colors(i, :), 'DisplayName', sprintf('%d Nulls Fit', numInterferers + 1 - i));
end
grid on;
xlabel('Distance between TX and RX (m)');
ylabel('BER');
title('BER vs. Distance with Different Numbers of Nulls Towards Interferers (Case 1)');
legend('show');
set(gca, 'YScale', 'log');
hold off;

% Plot BER results for Case 2
figure;
hold on;
for i = 1:numInterferers
    x = distances_m;
    y = squeeze(ber_results_case2(i, :, 1)); % Plot for first positional error std deviation
    std_dev = squeeze(ber_std_case2(i, :, 1));

    % Plot error bars for standard deviation
    errorbar(x, y, std_dev, 'o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), 'MarkerEdgeColor', colors(i, :), 'LineWidth', 1, 'CapSize', 5, 'DisplayName', sprintf('%d Nulls Raw Data', numInterferers + 1 - i));

    % Plot mean BER
    p = polyfit(x, y, 3);
    xx = linspace(min(x), max(x), 1000);
    yy = polyval(p, xx);
    plot(xx, yy, 'LineWidth', 2, 'Color', colors(i, :), 'DisplayName', sprintf('%d Nulls Fit', numInterferers + 1 - i));
end
grid on;
xlabel('Distance between TX and RX (m)');
ylabel('BER');
title('BER vs. Distance with Different Numbers of Nulls Towards Interferers (Case 2)');
legend('show');
set(gca, 'YScale', 'log');
hold off;

% Plot positions for all scenarios of interferer placements before and after positional errors
plot_all_interferer_positions(initial_tx_location_case1, 0, 'Simulation with Nulls (Case 1)', interferer_positions, positional_errors_std);
plot_all_interferer_positions(initial_tx_location_case2, 18, 'Simulation with Nulls (Case 2)', interferer_positions, positional_errors_std);

% Plot radiation patterns for TX, RX, and Interferer arrays
plot_radiation_patterns(numElements, numInterferers);

% 3D plot for varying positional errors
[X, Y] = meshgrid(distances_m, positional_errors_std);
figure;
surf(X, Y, squeeze(sinr_results_case1(1, :, :))');
xlabel('Distance between TX and RX (m)');
ylabel('Positional Error Std (m)');
zlabel('SINR (dB)');
legend('SINR');
title('3D Plot of SINR vs. Distance and Positional Error Std (Case 1)');
grid on;

[X, Y] = meshgrid(distances_m, positional_errors_std);
figure;
surf(X, Y, squeeze(sinr_results_case2(1, :, :))');
xlabel('Distance between TX and RX (m)');
ylabel('Positional Error Std (m)');
zlabel('SINR (dB)');
legend('SINR');
title('3D Plot of SINR vs. Distance and Positional Error Std (Case 2)');
grid on;

% Define varying nulls range
varying_nulls = 1:numInterferers;

% Prepare data for 3D plot
[X, Y] = meshgrid(distances_m, varying_nulls);
Z = squeeze(sinr_results_case1(:, :, 1)); % Adjust as needed for the correct SINR data

% Create 3D plot for SINR vs Distance and Number of Nulls
figure;
surf(X, Y, Z);
xlabel('Distance between TX and RX (m)');
ylabel('Number of Nulls');
zlabel('SINR (dB)');
title('3D Plot of SINR vs Distance and Number of Nulls (Case 1)');
colorbar;
legend('SINR');
grid on;

% Repeat for Case 2 if needed
Z_case2 = squeeze(sinr_results_case2(:, :, 1)); % Adjust as needed for the correct SINR data

figure;
surf(X, Y, Z_case2);
xlabel('Distance between TX and RX (m)');
ylabel('Number of Nulls');
zlabel('SINR (dB)');
title('3D Plot of SINR vs Distance and Number of Nulls (Case 2)');
colorbar;
legend('SINR');
grid on;

% Function Definitions

function [avgSINR_dB, avgBER] = run_simulation_with_nulls(numNulls, numInterferers, numIterations, initial_tx_location, initial_rx_location, interferer_positions, distances_m, positional_error_std)
    % System Parameters
    txPower_dBm = 24;
    carrierFreq = 5.9e9;
    channelBandwidth = 10e6;
    noiseFigures_dB = 15;
    lambda = 3e8 / carrierFreq;
    numElementsTX = 10;
    numElementsRX = 10;
    elementSpacing = lambda / 2;
    shadowingStdDev_dB = 4;

    % Mobility Parameters
    speedTX = 31.5 * 2.23694;  % Speed of TX vehicle in mph
    speedRX = 31.5 * 2.23694;  % Speed of RX vehicle in mph

    % Convert txPower from dBm to watts
    txPower_W = 10 ^ ((txPower_dBm - 30) / 10);

    % Create phased array ULA system objects for TX, RX, Interferers
    txArray = phased.ULA('NumElements', numElementsTX, 'ElementSpacing', elementSpacing);
    rxArray = phased.ULA('NumElements', numElementsRX, 'ElementSpacing', elementSpacing);

    % Preallocate arrays for SINR and BER values for each iteration
    SINR_dB_iterations = zeros(length(distances_m), length(noiseFigures_dB), numIterations);
    BER_iterations = zeros(length(distances_m), length(noiseFigures_dB), numIterations);

    for iter = 1:numIterations
        % Use a random set of pre-generated interferer positions for each iteration
        interferer_locations = interferer_positions{randi(length(interferer_positions))};

        % Introduce positional errors to the interferers
        interferer_locations = interferer_locations + positional_error_std * randn(size(interferer_locations));

        % Initialize positions
        tx_location = initial_tx_location;
        rx_location = initial_rx_location;

        % Update positions based on mobility models
        tx_location = update_position(tx_location, speedTX, 0);
        rx_location = update_position(rx_location, speedRX, 0);

        % Assume constant interference power (in dBm) for simplicity
        interferencePower_dBm = -20 + 10 * randn(1, numInterferers);

        % Calculate SINR and BER
        for j = 1:length(noiseFigures_dB)
            noiseFigure_dB = noiseFigures_dB(j);
            noisePower_W = 10 ^ ((-174 + 10 * log10(channelBandwidth) + noiseFigure_dB - 30) / 10);

            for i = 1:length(distances_m)
                distance = distances_m(i);
                pathLoss_dB = 20 * log10(4 * pi * distance / lambda);
                shadowing_dB = shadowingStdDev_dB * randn;
                totalPathLoss_dB = pathLoss_dB + shadowing_dB;
                totalReceivedPower_W = 0;
                totalInterferencePower_W = 0;

                % Calculate interference power
                for k = 1:numInterferers
                    interfererPos = interferer_locations(k, :);
                    interferenceDirection = atan2d(interfererPos(2), interfererPos(1) - rx_location(1));

                    % Calculate steering vector for interferer
                    interfererSteeringVector = steervec(getElementPosition(rxArray), interferenceDirection);

                    % Optimize RX array weights using LMS
                    rxWeights = lms_weights(rxArray, interfererSteeringVector, numElementsRX);

                    % Normalize weights
                    rxWeights = rxWeights / norm(rxWeights);

                    % Ensure dimensions match for dot product
                    minLen = min(length(rxWeights), length(interfererSteeringVector));
                    effectiveInterferencePower_W = 10 ^ ((interferencePower_dBm(k) - 30) / 10) * abs(dot(rxWeights(1:minLen), interfererSteeringVector(1:minLen))) ^ 2;
                    totalInterferencePower_W = totalInterferencePower_W + effectiveInterferencePower_W;
                end

                % Generate random multipaths
                numMultipaths = 10;
                multipathDelays = rand(1, numMultipaths) * 1e-6; % Random delays within 1 microsecond
                multipathGains = 10 .^ (randn(1, numMultipaths) * 0.1); % Random gains with a small deviation

                % Calculate received power for each multipath
                for n = 1:numMultipaths
                    multipathGain = multipathGains(n);
                    multipathDelay = multipathDelays(n);
                    rayPhase = exp(1i * 2 * pi * carrierFreq * multipathDelay);
                    fading = rayleigh_fading();

                    rayPathLoss_dB = totalPathLoss_dB - 10 * log10(multipathGain);
                    rayPathLoss = 10 ^ (-rayPathLoss_dB / 10);

                    % Optimize TX array weights using LMS
                    rxDirection = atan2d(0, distance);
                    txSteeringVector = steervec(getElementPosition(txArray), rxDirection);
                    txWeights = lms_weights(txArray, txSteeringVector, numElementsTX);

                    % Normalize weights
                    txWeights = txWeights / norm(txWeights);

                    % Adjust size of txWeights to match rxWeights for dot product
                    minLen = min(length(txWeights), length(rxWeights));
                    receivedPower_W = txPower_W * abs(dot(txWeights(1:minLen), rxWeights(1:minLen))) ^ 2 * rayPathLoss * fading * abs(rayPhase) ^ 2;
                    totalReceivedPower_W = totalReceivedPower_W + receivedPower_W;
                end

                % Calculate SINR and BER
                SINR = 10 * log10(totalReceivedPower_W / (totalInterferencePower_W + noisePower_W));
                SINR_dB_iterations(i, j, iter) = SINR;

                SINR_linear = 10^(SINR/10);

                % Ensure SINR_linear is real and non-negative before passing to qfunc
                SINR_linear_real = real(SINR_linear);
                SINR_linear_real(SINR_linear_real < 0) = 0;

                BER_iterations(i, j, iter) = qfunc(sqrt(2 * SINR_linear_real));
            end
        end
    end

    % Average SINR and BER over all iterations
    avgSINR_dB = mean(SINR_dB_iterations, 3);
    avgBER = mean(BER_iterations, 3);
end

function plot_all_interferer_positions(tx_location, lane_offset_RX, title_desc, interferer_positions, positional_errors_std)
    % Plot positions of TX, RX, and Interferers for all scenarios
    figure;
    hold on;
    grid on;
    plot(tx_location(1), tx_location(2), 'ko', 'MarkerFaceColor', 'k', 'DisplayName', 'TX');

    % Plot RX positions
    distances_m = 10:10:400;
    for i = 1:length(distances_m)
        rx_location = [distances_m(i), lane_offset_RX];
        plot(rx_location(1), rx_location(2), 'b*');
    end

    % Plot all interferer positions for each scenario
    for i = 1:length(interferer_positions)
        interferer_locations = interferer_positions{i};

        % Plot original interferer positions
        plot(interferer_locations(:,1), interferer_locations(:,2), 'r^', 'DisplayName', 'Interferers (Original)');

        % Apply positional errors
        interferer_locations_with_error = interferer_locations + positional_errors_std(end) * randn(size(interferer_locations));

        % Plot interferer positions with positional errors
        plot(interferer_locations_with_error(:,1), interferer_locations_with_error(:,2), 'm^', 'DisplayName', 'Interferers (With Errors)');
    end

    xlabel('Distance (m)');
    ylabel('Lane Position (m)');
    title(['TX, RX, and Interferer Positions for ' title_desc]);
    legend('TX', 'RX Locations', 'Interferers (Original)', 'Interferers (With Errors)');
    axis equal;
    xlim([-20, max(distances_m) + 20]);
    ylim([-50, 50]);
    hold off;
end

function weights = lms_weights(array, steeringVector, numElements)
    % LMS beamforming weights calculation
    % Parameters
    mu = 0.01; % Step size for LMS algorithm
    desired_signal = 1; % Desired signal level

    % Initialize weights
    weights = ones(numElements, 1) / numElements;

    % Ensure steeringVector is a column vector for each element
    if size(steeringVector, 2) ~= 1
        steeringVector = steeringVector';
    end

    % Ensure weights is a column vector
    if isrow(weights)
        weights = weights';
    end

    % Adapt weights using LMS algorithm
    for n = 1:100 % Number of iterations for adaptation
        output = weights' * steeringVector;
        error = desired_signal - output;

        % Update weights
        weights = weights + mu * conj(error) .* steeringVector;
    end

    % Normalize weights
    weights = weights / norm(weights);
end

function new_position = update_position(position, speed, direction)
    % Update position based on speed and direction
    dt = 1; % time step in seconds
    new_position = position + speed * dt * [cosd(direction), sind(direction)];
end

function interferer_positions = generate_all_interferer_positions(numInterferers, min_distance, max_distance, step_distance, lane_positions)
    % Generate all possible positions for the interferers
    interferer_positions = {};
    for i = min_distance:step_distance:max_distance
        for lane = lane_positions
            interferer_positions{end+1} = [i * ones(numInterferers, 1), lane * ones(numInterferers, 1)];
        end
    end
end

function fading = rayleigh_fading()
    % Generate Rayleigh fading
    fading = sqrt(0.5) * (randn + 1i * randn);
end

function plot_radiation_patterns(numElements, numInterferers)
    carrierFreq = 5.9e9;
    lambda = 3e8 / carrierFreq;
    elementSpacing = lambda / 2;
    azimuthAngles = -180:1:180;  % Adjust to match the pattern resolution

    txArray = phased.ULA('NumElements', numElements, 'ElementSpacing', elementSpacing);
    rxArray = phased.ULA('NumElements', numElements, 'ElementSpacing', elementSpacing);

    initialWeights = ones(numElements, 1) / numElements;
    interfererAngles = linspace(-90, 90, numInterferers);
    interfererSteeringVectors = steervec(getElementPosition(rxArray), interfererAngles);

    % Debug statements
    disp('Interferer Steering Vectors Size:');
    disp(size(interfererSteeringVectors));

    optimizedWeights = lms_weights(rxArray, interfererSteeringVectors(:, 1), numElements); % Ensure correct dimensions

    % Debug statement
    disp('Optimized Weights Size:');
    disp(size(optimizedWeights));

    % Generate patterns
    patternInitialTx = pattern(txArray, carrierFreq, azimuthAngles, 'PropagationSpeed', 3e8, 'Weights', initialWeights, 'Type', 'directivity');
    patternTx = pattern(txArray, carrierFreq, azimuthAngles, 'PropagationSpeed', 3e8, 'Weights', optimizedWeights, 'Type', 'directivity');
    patternInitialRx = pattern(rxArray, carrierFreq, azimuthAngles, 'PropagationSpeed', 3e8, 'Weights', initialWeights, 'Type', 'directivity');
    patternRx = pattern(rxArray, carrierFreq, azimuthAngles, 'PropagationSpeed', 3e8, 'Weights', optimizedWeights, 'Type', 'directivity');

    % Print debug info
    disp('patternInitialTx length:');
    disp(length(patternInitialTx));
    disp('azimuthAngles length:');
    disp(length(azimuthAngles));

    % Ensure dimensions match
    assert(length(patternInitialTx) == length(azimuthAngles), 'Dimension mismatch: patternInitialTx and azimuthAngles');
    assert(length(patternTx) == length(azimuthAngles), 'Dimension mismatch: patternTx and azimuthAngles');
    assert(length(patternInitialRx) == length(azimuthAngles), 'Dimension mismatch: patternInitialRx and azimuthAngles');
    assert(length(patternRx) == length(azimuthAngles), 'Dimension mismatch: patternRx and azimuthAngles');

    % Plot initial and optimized radiation patterns
    figure;
    subplot(2, 1, 1);
    plot(azimuthAngles, patternInitialTx, 'DisplayName', 'Initial');
    hold on;
    plot(azimuthAngles, patternTx, 'DisplayName', 'Optimized');
    title('Radiation Pattern of TX Array');
    xlabel('Azimuth Angles (degrees)');
    ylabel('Directivity (dB)');
    legend('show');
    hold off;

    subplot(2, 1, 2);
    plot(azimuthAngles, patternInitialRx, 'DisplayName', 'Initial');
    hold on;
    plot(azimuthAngles, patternRx, 'DisplayName', 'Optimized');
    title('Radiation Pattern of RX Array');
    xlabel('Azimuth Angles (degrees)');
    ylabel('Directivity (dB)');
    legend('show');
    hold off;
end


function analyze_pattern(arrayName, pattern, az)
    % Ensure pattern and az are vectors of the same length
    if length(pattern) ~= length(az)
        error('The length of pattern and azimuth angles must be the same.');
    end

    % Find the main lobe
    [mainLobeValue, mainLobeIdx] = max(pattern);
    mainLobeAz = az(mainLobeIdx);

    % Find side lobes
    [sideLobes, sideLobeIdxs] = findpeaks(pattern);

    % Find nulls (using a threshold, e.g., -20 dB)
    nullThreshold = -20; % Example threshold for nulls
    nulls = find(pattern < nullThreshold);

    % Display results
    fprintf('%s Array:\n', arrayName);
    fprintf('Main Lobe: Azimuth = %d degrees, Directivity = %.2f dB\n', mainLobeAz, mainLobeValue);
    fprintf('Number of Side Lobes: %d\n', length(sideLobes));
    fprintf('Number of Nulls: %d\n', length(nulls));
    fprintf('Side Lobe Angles (degrees):\n');
    disp(az(sideLobeIdxs));
    fprintf('Null Angles (degrees):\n');
    disp(az(nulls));
end
