clear; close all; clc;

% Call the main function
SINR_simulation();

function SINR_simulation()
    parameters = set_parameters();
    iterations = 1000;

    % Define the range of radii for the interferers
    radius_values = 20:40:200;
    SINR_results = zeros(length(parameters.d_values), length(radius_values));  % Store average SINR values for every radius
    SINR_std_results = zeros(length(parameters.d_values), length(radius_values));  % Store standard deviation values for every radius

    for r_idx = 1:length(radius_values)
        [TX, RX, interferers, relative_velocity] = set_positions(parameters, radius_values(r_idx));
        SINR_values_matrix = zeros(length(parameters.d_values), iterations);  % Store SINR values for every iteration
        
        for it = 1:iterations      
            for idx = 1:length(parameters.d_values)
                RX(1) = parameters.d_values(idx);  % updating RX position
                SINR_values_matrix(idx, it) = calculateSINR(TX, RX, interferers, relative_velocity, parameters);
            end
        end
        SINR_avg_values = mean(SINR_values_matrix, 2);  % Average the SINR values across iterations
        SINR_std_values = std(SINR_values_matrix, 0, 2); % Compute standard deviation

        SINR_results(:, r_idx) = SINR_avg_values;  % Store the average SINR values for the current radius
        SINR_std_results(:, r_idx) = SINR_std_values;  % Store the standard deviation values for the current radius

        % Optional: Visualize positions for every radius (can be commented out if not needed)
        visualize_positions(TX, RX, interferers, parameters);
    end

    plot_results(parameters.d_values, SINR_results, SINR_std_results, radius_values); 
end

function parameters = set_parameters()
   parameters.P_s_tx = 1;
    parameters.wavelength = 0.3;
%     parameters.d0 = 10; %in meters
%     parameters.PL_d0 = 82.86; %free space path loss at 10 meters and an additional environmental margin of 15 dB to account for factors like shadowing and multipath effects
    parameters.d_spacing = parameters.wavelength / 2;    
    parameters.sigma2 = 0.01;
    parameters.pathLossExponent = 3;
    parameters.N_elements_tx = 8;
    parameters.N_elements_rx = 8;
    parameters.N_elements_interferer = 4;
    parameters.c = 3e8;  
    parameters.fc = 5.9e9;
    parameters.d_values = 10:10:400;
    parameters.interfererPowerRange = [0.1, 1]; % example range
    parameters.TX_height = 30; % in meters
    parameters.RX_height = 1.5; % in meters
    parameters.antennaGain = 3; % linear scale
    parameters.multipathExponent = 2; % Example
    parameters.DoA_signal = 30; % in degrees
    parameters.DoA_interferers = [45, 60, 90]; % in degrees for each interferer
    parameters.w = ones(parameters.N_elements_rx, 1); % Initial weights
    parameters.mu = 0.01; % Step size for LMS
    parameters.interferenceRange = 50;  % example range in meters
    parameters.relative_velocity = [40, 0, 0]; % Example relative velocity in m/s
    parameters.sigma_shadow = 8; % Shadowing standard deviation in dB
    parameters.multipath_delays = [0, 1e-6, 3e-6, 5e-6, 7e-6]; % 5 multipath components
    parameters.multipath_attenuation = [1, 0.7, 0.5, 0.3, 0.1];

    return;
end

function [TX, RX, interferers, relative_velocity] = set_positions(parameters, radius)
    TX = [0,0, parameters.TX_height];
    RX = [0,0, parameters.RX_height];
    n = 10; 
    angles = linspace(0, 2*pi, n+1);
    angles = angles(1:end-1);
    interferers = [radius * cos(angles)', radius * sin(angles)', ones(n, 1) * 1.5];  % added height for interferers    
    relative_velocity = parameters.relative_velocity; % Relative velocity vector
    return;
end

function [x, d] = generate_signals(TX, RX, interferers, parameters)
    % Number of antenna elements
    N = parameters.N_elements_rx;

    % Wavelength (needed for array factor calculation)
    lambda = parameters.wavelength;

    % Initialize the received signal vector and the desired signal
    x = zeros(N, 1);
    d = zeros(N, 1);

    % Generate the desired signal (assuming it comes from the direction of the TX)
    % Calculate the phase shift for each antenna element due to the DoA
    for n = 1:N
        d_phase_shift = 2 * pi / lambda * (n - 1) * parameters.d_spacing * cos(deg2rad(parameters.DoA_signal));
        array_factor = sin(N * pi * parameters.d_spacing / lambda * sin(deg2rad(parameters.DoA_signal))) / (N * sin(pi * parameters.d_spacing / lambda * sin(deg2rad(parameters.DoA_signal))));
        d(n) = array_factor * exp(1i * d_phase_shift); % Applying array factor
    end

    % Generate interfering signals
    for k = 1:length(interferers)
        % For each interferer, calculate the phase shift for each antenna element
        for n = 1:N
            k_mod = mod(k-1, length(parameters.DoA_interferers)) + 1;
            i_phase_shift = 2 * pi / lambda * (n - 1) * parameters.d_spacing * cos(deg2rad(parameters.DoA_interferers(k_mod)));
            array_factor = sin(N * pi * parameters.d_spacing / lambda * sin(deg2rad(parameters.DoA_interferers(k_mod)))) / (N * sin(pi * parameters.d_spacing / lambda * sin(deg2rad(parameters.DoA_interferers(k_mod)))));
            x(n) = x(n) + array_factor * exp(1i * i_phase_shift); % Adding the interferer's contribution with array factor
        end
    end

    % Add the desired signal to the received signal vector
    x = x + d;

    % Add noise to the received signal
    noise_variance = parameters.sigma2;
    noise = sqrt(noise_variance/2) * (randn(N, 1) + 1i * randn(N, 1));
    x = x + noise;
end

function w = update_weights_LMS(w, x, d, mu)
    if length(w) ~= length(x)
        disp(['Error: Length of weight vector (', num2str(length(w)), ') does not match length of signal vector (', num2str(length(x)), ').']);
        error('Length of weight vector and signal vector must be the same');
    end

    y = w' * x; % Beamformed signal
    e = d - y; % Error signal (should be a scalar)

    % Element-wise multiplication and update
    w = w + mu * e .* x; % Note the use of .*
    return;
end

function SINR = calculateSINR(TX, RX, interferers, relative_velocity, parameters)
    % Placeholder for beamforming gain, antenna array, etc.
    G_s = parameters.N_elements_tx * parameters.N_elements_rx;
    G_i = ones(size(interferers, 1), 1);  % Example interferer gains

    % Include adaptive beamforming using LMS
    [x, d] = generate_signals(TX, RX, interferers, parameters);
    parameters.w = update_weights_LMS(parameters.w, x, d, parameters.mu); 
    
    % Compute Multipath effect
    multipathFactor = 1 / (1 + parameters.multipathExponent);
    
    % Calculate path loss
    pathLossDesired = path_loss(TX, RX, parameters);
    pathLossInterferers = arrayfun(@(idx) path_loss(interferers(idx, :), RX, parameters), 1:size(interferers, 1));
    
    % Signal and Interference Power
    signalPower = parameters.P_s_tx * G_s / pathLossDesired;
    interferencePowers = parameters.interfererPowerRange(1) + (parameters.interfererPowerRange(2) - parameters.interfererPowerRange(1)) * rand(size(interferers, 1), 1);  % random interference powers within range
    interferencePower = sum(interferencePowers .* G_i ./ pathLossInterferers);

       % Noise Power
    noisePower = parameters.sigma2;

        % Incorporate Doppler Shift
    f_doppler = parameters.fc * norm(relative_velocity) / parameters.c;
    doppler_phase_shift = 2 * pi * f_doppler * (1/parameters.fc); % Assuming t = 1/f_c for simplicity

    % Incorporate Shadowing
    shadowing_db = normrnd(0, parameters.sigma_shadow);
    shadowing_factor = 10^(shadowing_db / 10);

    % Incorporate Multipath
    multipath_factor = 0;
    for i = 1:length(parameters.multipath_delays)
        multipath_phase_shift = 2 * pi * parameters.fc * parameters.multipath_delays(i);
        multipath_factor = multipath_factor + parameters.multipath_attenuation(i) * exp(1i * multipath_phase_shift);
    end

    % Adjust signal and interference powers for Doppler, Shadowing, and Multipath
    signalPower = signalPower * shadowing_factor * abs(multipath_factor);
    interferencePower = interferencePower * shadowing_factor; % Assuming same shadowing factor for simplicity
    
    % Total Interference Power
    totalInterferencePower = sum(interferencePower);
    
     % Compute SINR
    SINR = signalPower / (totalInterferencePower + noisePower);

     return;
end

function visualize_positions(TX, RX, interferers, parameters)
    figure;
    
    % Plot Transmitter
    plot3(TX(1), TX(2), TX(3), '^', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'TX');
    hold on;

    % Plot Receiver
    plot3(RX(1), RX(2), RX(3), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'RX');

    % Plot Interferers
    for i = 1:size(interferers, 1)
        plot3(interferers(i, 1), interferers(i, 2), interferers(i, 3), '*', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', ['Interferer ' num2str(i)]);
        % Visualize Interference Range (using a sphere for demonstration)
        [x, y, z] = sphere;
        surf(x*parameters.interferenceRange + interferers(i, 1), y*parameters.interferenceRange + interferers(i, 2), z*parameters.interferenceRange + interferers(i, 3), 'FaceAlpha', 0.1, 'FaceColor', 'r', 'EdgeColor', 'none');
    end

    % Display beamforming direction
    quiver3(TX(1), TX(2), TX(3), RX(1)-TX(1), RX(2)-TX(2), RX(3)-TX(3), 0, 'LineWidth', 1.5, 'Color', 'b', 'DisplayName', 'Beamforming Direction', 'MaxHeadSize', 2);
    
    % Other Visualization Parameters
    title('3D Visualization of TX, RX and Interferers');
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    zlabel('Height (m)');
    grid on;
    legend show;
    axis equal;
    view(45, 25);  % optimal azimuth and elevation for view
    hold off;
end


function L = path_loss(TX, RX, parameters)
    d = norm(TX(1:2) - RX(1:2));  % horizontal distance
    delta_h = abs(TX(3) - RX(3));  % height difference
    L = parameters.antennaGain / (d^parameters.pathLossExponent + delta_h^2);  % simple path loss with height consideration

    return;
end

function plot_results(d_values, SINR_results, SINR_std_results, radius_values)
    figure;
    colors = jet(size(SINR_results, 2)); % Create color array

    % Loop over each radius value
    for r_idx = 1:size(SINR_results, 2)
        % Convert SINR values and their standard deviations to dB
        SINR_dB = 10 * log10(SINR_results(:, r_idx));
        SINR_std_dB = 10 * log10(SINR_std_results(:, r_idx));

        % Prepare data for shaded area (mean ± standard deviation)
        x = [d_values, fliplr(d_values)];
        y_upper = SINR_dB + SINR_std_dB;
        y_lower = SINR_dB - SINR_std_dB;
        y = [y_upper', fliplr(y_lower')];

        % Plot the shaded area
        fill(x, y, colors(r_idx, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        hold on;

        % Plot the mean SINR values
        plot(d_values, SINR_dB, 'LineWidth', 1.5, 'Color', colors(r_idx, :), 'DisplayName', ['Radius = ' num2str(radius_values(r_idx))]);
    end

    grid on;
    xlabel('Distance (d) between TX and RX (m)');
    ylabel('SINR (dB)');
    title('SINR vs. Distance between TX and RX for different interferer radii');
    legend show;
    hold off;
%     % Displaying SINR_values array in dB
%     SINR_dB = 10*log10(SINR_results);
%     disp('SINR values in dB for the given distances and radii:');
%     disp(SINR_dB);
    return;
end