clear;

% Call the main function
SINR_simulation();

function SINR_simulation()
    parameters = set_parameters();
    n_values = 2:20;

    % Additional storage for standard deviations and averages
    SINR_std_values_random = zeros(1, length(n_values));
    SINR_avg_values_random = zeros(1, length(n_values));
    SINR_theoretical_values_random = zeros(1, length(n_values));

    iterations = 1000;

    for n_idx = 1:length(n_values)
        n = n_values(n_idx);
        
        % Initialize dynamic positions
        [TX, RX, interferers_random] = initialize_dynamic_positions(parameters, n);

        SINR_values_random = zeros(1, iterations);
        SINR_values_theoretical = zeros(1, iterations);

        for it = 1:iterations
            % Update positions for V2V movement
            [TX, RX, interferers_random] = update_positions(TX, RX, interferers_random, parameters);
           
            % Calculate SINR with updated positions
            SINR_values_random(it) = calculateSINR(TX, RX, interferers_random, parameters);
             SINR_values_theoretical(it) = calculateTheoreticalSINR(TX, RX, interferers_random, parameters);
        end
        
        SINR_avg_values_random(n_idx) = mean(SINR_values_random);
        SINR_std_values_random(n_idx) = std(SINR_values_random);
        SINR_theoretical_values_random(n_idx) = mean(SINR_values_theoretical); 
    end
    
     plot_results(n_values, SINR_avg_values_random, SINR_theoretical_values_random, SINR_std_values_random);

     % Display final SINR values as arrays
    disp('Final Simulated SINR Values (dB):');
    disp(10*log10(SINR_avg_values_random));
    disp('Final Theoretical SINR Values (dB):');
    disp(10*log10(SINR_theoretical_values_random));

    % Add visualization at the end of each iteration
            visualize_positions(TX, RX, interferers_random);
            pause(0.1);
            
  % Add 2D visualization at the end of each iteration
            visualize_positions_2D(TX, RX, interferers_random);
            pause(0.1);  % Pauses for 0.1 seconds for visualization  
end

function parameters = set_parameters()
    parameters.P_s_tx = 1;
    parameters.sigma2 = 0.01;
    parameters.pathLossExponent = 3.5;
    parameters.N_elements_tx = 8;
    parameters.N_elements_rx = 8;
    parameters.N_elements_interferer = 4;
    parameters.wavelength = 0.3; 
    parameters.c = 3e8;  
    parameters.fc = 5.9e9;
    parameters.d_values = 10:10:400;
    parameters.interfererPowerRange = [0.5, 1.5]; % example range
    parameters.TX_height = 30; % in meters
    parameters.RX_height = 1.5; % in meters
    parameters.antennaGain = 1; % linear scale
    parameters.multipathExponent = 2; % Example
    parameters.interferenceRange = 50;
    return;
end

function SINR_theoretical = calculateTheoreticalSINR(TX, RX, interferers, parameters)
    G_s = parameters.N_elements_tx * parameters.N_elements_rx; % Gain for signal
    P_s = parameters.P_s_tx; % Signal Power

    % Dynamically calculate total interference power based on current positions
    total_interference_power = 0;
    for i = 1:size(interferers, 1)
        distance = norm(interferers(i, 1:2) - RX(1:2));
        L_i = distance^(-parameters.pathLossExponent);
        P_i = parameters.interfererPowerRange(1) + (parameters.interfererPowerRange(2) - parameters.interfererPowerRange(1)) * rand; % Random interference power within range
        total_interference_power = total_interference_power + P_i / L_i;
    end
    
    sigma2 = parameters.sigma2; % Noise power
    SINR_theoretical = G_s * P_s / (G_s * total_interference_power + sigma2);
    return;
end


function [TX, RX, interferers] = initialize_dynamic_positions(parameters, n)
     TX = [0, 0, parameters.TX_height];
     RX = [0, 0, parameters.RX_height];

    % Setting bounds for random placement of interferers
    sideLength = 200;  % You can adjust this value as per your requirements

    % Randomly placing interferers within the defined square region
    x_coords = (rand(n, 1) - 0.5) * sideLength;
    y_coords = (rand(n, 1) - 0.5) * sideLength;
    heights = ones(n, 1) * 1.5;  % Setting height for interferers

    interferers = [x_coords, y_coords, heights];
    return;
end

function [TX, RX, interferers] = update_positions(TX, RX, interferers, parameters)
    % Update positions based on movement
    speed = 10; % speed in m/s
    timestep = 1; % time step in seconds

    % Update TX and RX positions
    TX(1) = TX(1) + speed * timestep;
    RX(1) = RX(1) + speed * timestep;

    % Update interferers' positions
    for i = 1:size(interferers, 1)
        interferers(i, 1) = interferers(i, 1) + speed * timestep;
    end
    return;
end

function SINR = calculateSINR(TX, RX, interferers, parameters)
    G_s = parameters.N_elements_tx * parameters.N_elements_rx; % Gain for signal
    G_i = G_s * ones(size(interferers, 1), 1);  % Same gain for interferers as the signal
     
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

   % Total Interference Power
    totalInterferencePower = sum(interferencePower);

    % Compute SINR
    SINR = signalPower / (totalInterferencePower + noisePower);    return;
end

function L = path_loss(TX, RX, parameters)
    d = norm(TX(1:2) - RX(1:2));  % horizontal distance
    delta_h = abs(TX(3) - RX(3));  % height difference
    L = parameters.antennaGain / (d^parameters.pathLossExponent + delta_h^2);  % simple path loss with height consideration    return;
end

function visualize_positions(TX, RX, interferers)
    figure;
    
    % Plot Transmitter
    plot3(TX(1), TX(2), TX(3), '^', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'TX');
    hold on;

    % Plot Receiver
    plot3(RX(1), RX(2), RX(3), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'RX');

    % Plot Interferers
    for i = 1:size(interferers, 1)
        plot3(interferers(i, 1), interferers(i, 2), interferers(i, 3), '*', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', ['Interferer ' num2str(i)]);
    end

    % Other Visualization Parameters
    title('3D Visualization of TX, RX, and Interferers');
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    zlabel('Height (m)');
    grid on;
    legend show;
    axis equal;
    view(45, 25);  % Optimal azimuth and elevation for view
    hold off;
end

function plot_results(n_values, SINR_values_random, SINR_theoretical_values_random, SINR_std_random)
    % Modified plot_results function for simplified V2V model
    figure;

    % SINR plots with shaded standard deviation for Random Placement
    x = [n_values, fliplr(n_values)];
    y = [10*log10(SINR_values_random + SINR_std_random), fliplr(10*log10(SINR_values_random - SINR_std_random))];
    fill(x, y, [0.9 0.9 0.9], 'HandleVisibility', 'off');
    hold on;
    plot(n_values, 10*log10(SINR_values_random), 'r', 'DisplayName', 'Simulated Random');

    % SINR theoretical values for Random Placement
    plot(n_values, 10*log10(SINR_theoretical_values_random), 'b', 'DisplayName', 'Theoretical Random');

    % Labels and titles
    xlabel('Number of Interferers');
    ylabel('SINR (dB)');
    title('SINR versus Number of Interferers in V2V Scenario');
    legend show;
    grid on;
    hold off;
end

function visualize_positions_2D(TX, RX, interferers)
    figure;
    
    % Plot Transmitter
    plot(TX(1), TX(2), '^', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'TX');
    hold on;

    % Plot Receiver
    plot(RX(1), RX(2), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'RX');

    % Plot Interferers
    plot(interferers(:, 1), interferers(:, 2), '*', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Interferers');

    % Other Visualization Parameters
    title('2D Visualization of TX, RX, and Interferers');
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    grid on;
    legend show;
    axis equal;
    view(0, 90);  % View from above
    hold off;
end