% ==============================================================================
% 3GPP TDL-A Channel Dataset Generator for OFDM System
% Authors: Mahfoud Faraj Abdulmolla, Osama Abusaid
% Faculty of Engineering, University of Tripoli
% Project: FortiTranM-FiLM Channel Estimation
% ==============================================================================
%
% Generates realistic high-mobility OFDM channel matrices conforming to 5G NR
% physical-layer specifications across configurable SNR, Doppler, and delay spread.

function generate_dataset(num_samples, output_dir, mode, fixed_val)
    if nargin < 1, num_samples = 100000; end
    if nargin < 2, output_dir = 'dataset'; end
    if nargin < 3, mode = 'randomized'; end
    if nargin < 4, fixed_val = 0; end

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % 5G NR OFDM Numerology
    scs = 15e3;                  % Subcarrier spacing: 15 kHz
    n_sc = 120;                  % Number of active subcarriers (10 RBs)
    n_sym = 14;                  % Number of OFDM symbols per frame
    carrier_freq = 3.5e9;        % Carrier frequency: 3.5 GHz
    sample_rate = 3.84e6;        % Sampling frequency: 3.84 MHz
    
    % Pilot indices (3rd and 12th symbols, every 3 subcarriers)
    pilot_symbols = [3, 12];     % 1-based index (MATLAB)
    pilot_subcarriers = 2:3:119; % 40 subcarriers

    fprintf('Generating %d realizations in %s mode...\n', num_samples, mode);

    for idx = 1:num_samples
        % Channel parameters selection
        switch mode
            case 'randomized'
                snr_db = rand() * 30;           % SNR in [0, 30] dB
                ds = (rand() * 290 + 10) * 1e-9;% Delay spread in [10, 300] ns
                doppler = rand() * 1000;        % Max Doppler in [0, 1000] Hz
            case 'fixed_snr'
                snr_db = fixed_val;
                ds = 100e-9;
                doppler = 500;
            case 'fixed_ds'
                snr_db = 15;
                ds = fixed_val * 1e-9;
                doppler = 500;
            case 'fixed_doppler'
                snr_db = 15;
                ds = 100e-9;
                doppler = fixed_val;
        end

        % Setup 3GPP TDL-A Channel Profile
        tdl = nrTDLChannel;
        tdl.DelayProfile = 'TDL-A';
        tdl.DelaySpread = ds;
        tdl.MaximumDopplerShift = doppler;
        tdl.SampleRate = sample_rate;
        tdl.NumTransmitAntennas = 1;
        tdl.NumReceiveAntennas = 1;

        % Generate Channel Impulse Response
        [~, pathGains] = tdl();
        pathFilters = getPathFilters(tdl);

        % Compute Frequency Response Grid (120 subcarriers x 14 symbols)
        H_ideal = nrPerfectChannelEstimate(pathGains, pathFilters, n_sc, scs, 0);
        H_ideal = H_ideal(1:n_sc, 1:n_sym);

        % Simulate Pilot Transmission with Noise
        pilot_mask = false(n_sc, n_sym);
        pilot_mask(pilot_subcarriers, pilot_symbols) = true;
        
        noise_var = 10^(-snr_db / 10);
        noise = sqrt(noise_var / 2) * (randn(n_sc, n_sym) + 1i * randn(n_sc, n_sym));
        
        H_ls = zeros(n_sc, n_sym);
        H_ls(pilot_mask) = H_ideal(pilot_mask) + noise(pilot_mask);

        % Save Sample
        sample_path = fullfile(output_dir, sprintf('sample_%06d.mat', idx));
        H = cat(3, H_ideal, H_ls);
        save(sample_path, 'H', 'snr_db', 'ds', 'doppler', '-v7');

        if mod(idx, 5000) == 0
            fprintf('  Generated %d / %d samples (%.1f%%)\n', idx, num_samples, idx/num_samples*100);
        end
    end

    fprintf('Generation complete! Saved to %s\n', output_dir);
end
