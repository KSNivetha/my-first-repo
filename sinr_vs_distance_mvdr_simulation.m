clear all; clc; close all;

% Number of iterations for averaging
numIterations = 10;

% Setup for Case 1: TX and RX in opposite lanes
[SINR_dB_case1, BER_case1] = run_simulation(10, 'Case 1: TX and RX in opposite lanes with interferers across both lanes', numIterations);

% Setup for Case 2: TX and RX in the same lane
[SINR_dB_case2, BER_case2] = run_simulation(0, 'Case 2: TX and RX in the same lane with interferers in the opposite lane', numIterations);

function [avgSINR_dB, avgBER] = run_simulation(lane_offset_RX, title_desc, numIterations)
    % System Parameters
    txPower_dBm = 24;
    carrierFreq = 5.9e9;
    channelBandwidth = 10e6;
    noiseFigures_dB = 9:3:24;
    lambda = 3e8 / carrierFreq;
    numInterferers = 10;
    tx_location = [0, 0];
    lane_width = 18;  % Standard lane width in meters
    numElementsTX = 8;
    numElementsRX = 4;
    numElementsInterferer = 16;
    elementSpacing = lambda/2;
    shadowingStdDev_dB = 4;

    % Mobility Parameters
    speedTX = 31.5 * 2.23694;  % Speed of TX vehicle in mph
    speedRX = 31.5 * 2.23694;  % Speed of RX vehicle in mph

    % Convert txPower from dBm to watts
    txPower_W = 10^((txPower_dBm - 30)/10);

    % Create phased array ULA system objects for TX, RX, Interferers
    txArray = phased.ULA('NumElements', numElementsTX, 'ElementSpacing', elementSpacing);
    rxArray = phased.ULA('NumElements', numElementsRX, 'ElementSpacing', elementSpacing);
    interfererArray = phased.ULA('NumElements', numElementsInterferer, 'ElementSpacing', elementSpacing);

    % Distance range from TX to RX
    distances_m = 10:10:400;

    % Adaptive Beamforming Assumed Gains
    interferenceReduction_dB = 10;

    % Preallocate array for SINR and BER values for each iteration
    SINR_dB_iterations = zeros(length(distances_m), length(noiseFigures_dB), numIterations);
    BER_iterations = zeros(length(distances_m), length(noiseFigures_dB), numIterations);

    for iter = 1:numIterations
        % Assume constant interference power (in dBm) for simplicity
        interferencePower_dBm = -20 + 10 * randn(1, numInterferers);

        % Define locations for interferers randomly across both lanes
        if lane_offset_RX == 0  % Case 2, interferers only in opposite lane
            interferer_locations = [rand(numInterferers, 1) * max(distances_m), repmat(lane_width, numInterferers, 1)];
        else  % Case 1, interferers across both lanes
            interferer_locations = [rand(numInterferers, 1) * max(distances_m), lane_width * round(rand(numInterferers, 1))];
        end

        % Calculate SINR and BER
        for j = 1:length(noiseFigures_dB)
            noiseFigure_dB = noiseFigures_dB(j);
            noisePower_W = 10^((-174 + 10*log10(channelBandwidth) + noiseFigure_dB - 30)/10);

            for i = 1:length(distances_m)
                distance = distances_m(i);
                pathLoss_dB = 20*log10(4*pi*distance/lambda);
                shadowing_dB = shadowingStdDev_dB * randn;
                totalPathLoss_dB = pathLoss_dB + shadowing_dB;
                totalReceivedPower_W = 0;
                totalInterferencePower_W = 0;

                % Calculate interference power
                for k = 1:numInterferers
                    interfererPos = interferer_locations(k, :);
                    interferenceDirection = atan2d(interfererPos(2) - lane_offset_RX, interfererPos(1));

                    % Calculate steering vector for interferer
                    interfererSteeringVector = steervec(getElementPosition(rxArray), [interferenceDirection; 0]);

                    % Optimize RX array weights using MVDR
                    rxWeights = mvdr_weights(interfererSteeringVector, noisePower_W);

                    effectiveInterferencePower_W = 10^((interferencePower_dBm(k) - interferenceReduction_dB - 30)/10) * abs(dot(rxWeights, interfererSteeringVector))^2;
                    totalInterferencePower_W = totalInterferencePower_W + effectiveInterferencePower_W;
                end

                % Calculate received power for each cluster and ray
                for n = 1:3 % Number of clusters
                    clusterGain_dB = [-1, -3, -5];
                    for m = 1:3 % Number of rays per cluster
                        rayGain_dB = clusterGain_dB(m) - 5 * rand(1);
                        rayPhase = exp(1i * (10 * randn(1)));
                        fading = gamrnd(4, 1/4);

                        rayPathLoss_dB = totalPathLoss_dB - rayGain_dB;
                        rayPathLoss = 10^(-rayPathLoss_dB / 10);

                        % Optimize TX array weights using MVDR
                        rxDirection = atan2d(lane_offset_RX, distance);
                        txSteeringVector = steervec(getElementPosition(txArray), [rxDirection; 0]);
                        txWeights = mvdr_weights(txSteeringVector, noisePower_W);

                        % Adjust size of txWeights to match rxWeights for dot product
                        if length(txWeights) > length(rxWeights)
                            txWeights = txWeights(1:length(rxWeights));
                        elseif length(txWeights) < length(rxWeights)
                            rxWeights = rxWeights(1:length(txWeights));
                        end

                        receivedPower_W = txPower_W * abs(dot(txWeights, rxWeights))^2 * rayPathLoss * fading * abs(rayPhase)^2;
                        totalReceivedPower_W = totalReceivedPower_W + receivedPower_W;
                    end
                end

                % Calculate SINR and BER
                SINR = 10 * log10(totalReceivedPower_W / (totalInterferencePower_W + noisePower_W));
                SINR_dB_iterations(i, j, iter) = SINR;

                SINR_linear = 10^(SINR/10);
                BER_iterations(i, j, iter) = qfunc(sqrt(2 * SINR_linear));
            end
        end
    end

    % Average SINR and BER over all iterations
    avgSINR_dB = mean(SINR_dB_iterations, 3);
    avgBER = mean(BER_iterations, 3);

    % Plot SINR vs. Distance
    figure;
    hold on;
    for j = 1:length(noiseFigures_dB)
        x = distances_m;
        y = avgSINR_dB(:, j);
        pp = spline(x, y);
        xx = linspace(min(x), max(x), 1000);
        yy = ppval(pp, xx);
        plot(xx, yy, 'LineWidth', 2);
    end
    grid on;
    xlabel('Distance between TX and RX (m)');
    ylabel('SINR (dB)');
    title(['SINR vs. Distance with Spline Fitting for ' title_desc]);
    legend(arrayfun(@(x) sprintf('NF = %d dB', x), noiseFigures_dB, 'UniformOutput', false));
    ylim([-90, -45]);
    xlim([10,400]);
    hold off;

    % Plot BER vs. Distance
    figure;
    hold on;
    for j = 1:length(noiseFigures_dB)
        x = distances_m;
        y = avgBER(:, j);
        pp = spline(x, y);
        xx = linspace(min(x), max(x), 1000);
        yy = ppval(pp, xx);
        plot(xx, yy, 'LineWidth', 2);
    end
    grid on;
    xlabel('Distance between TX and RX (m)');
    ylabel('BER');
    title(['BER vs. Distance with Spline Fitting for ' title_desc]);
    legend(arrayfun(@(x) sprintf('NF = %d dB', x), noiseFigures_dB, 'UniformOutput', false));
    ylim([0.485, 0.505]);
    xlim([10,400]);
    set(gca, 'YScale', 'log');
    hold off;

    % Visualization setup for TX, RX, and interferers
    figure;
    hold on;
    grid on;
    plot(tx_location(1), tx_location(2), 'ko', 'MarkerFaceColor', 'k');
    plot(interferer_locations(:,1), interferer_locations(:,2), 'r^', 'MarkerFaceColor', 'r');

    for distance = distances_m
        rx_location = [distance, lane_offset_RX];
        plot(rx_location(1), rx_location(2), 'b*');
    end

    xlabel('Distance (m)');
    ylabel('Lane Position (m)');
    title(title_desc);
    legend('TX', 'Interferers', 'RX Locations');
    axis equal;
    xlim([-20, max(distances_m) + 20]);
    ylim([-50, 50]);

    % 3D Visualization
    figure;
    surf(noiseFigures_dB, distances_m, avgSINR_dB);
    xlabel('Noise Figure (dB)');
    ylabel('Distance (m)');
    zlabel('SINR (dB)');
    title(['SINR vs. Distance and Noise Figure with Adaptive Beamforming and Clustered Multipath for ' title_desc]);
    colorbar;

    hold off;

    % Annotations for better understanding
    annotation('textbox', [.15 .6 .3 .3], 'String', sprintf('Tx Power: %d dBm\nCarrier Freq: %.1f GHz\nNoise Figure: Varies 9-24 dB\nTX ULA Elements: %d\nRX ULA Elements: %d\nInterferer ULA Elements: %d\nIntf. Reduction: %d dB\nNakagami-m Factor: %d', txPower_dBm, carrierFreq/1e9, numElementsTX, numElementsRX, numElementsInterferer, interferenceReduction_dB, 4), 'EdgeColor', 'black', 'BackgroundColor', 'white');
end

function weights = mvdr_weights(steeringVector, noisePower)
    % MVDR beamforming weights calculation
    R = steeringVector * steeringVector' + noisePower * eye(size(steeringVector, 1));
    weights = (R \ steeringVector) / (steeringVector' / R * steeringVector);
end
