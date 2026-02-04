clc; clear; close all;

%% ======== UNIFIED SYSTEM COMPARISON =======
% Compares OCDM, AFDM, and OTFS over NTN channels
% All systems use:
% - Same channel model
% - Same SNR range
% - Same modulation (16-QAM)
% - Same number of symbols (256)
% - LMMSE and MMSE-SD detectors

%% ================= COMMON PARAMETERS ==========
%================================================
params = struct();

% System dimensions
params.K = 16;
params.L = 16;
params.N = params.K * params.L;  % 256 symbols

% Modulation
params.bits_in_sym = 4;
params.M = 2^params.bits_in_sym;  % 16-QAM

% Physical parameters
params.fc = 2.55e9;               % Carrier frequency
params.df = 15e3;                 % Subcarrier spacing  15 kHz 
params.terminal_velocity = 500*5/18;  % 500 km/h in m/s
params.c = physconst('lightspeed');
params.max_doppler = params.terminal_velocity/params.c * params.fc*0.5;

% Simulation parameters
params.num_iter = 300;            % Monte Carlo iterations
params.SNR_dB = 0:4:20;           % SNR range
params.SNR = 10.^(params.SNR_dB/10);
params.channel_type = 'NTN_D';

% Display parameters
fprintf('=== UNIFIED NTN SYSTEM COMPARISON ===\n');
fprintf('Systems: OCDM, AFDM, OTFS\n');
fprintf('Symbols: %d, Modulation: %d-QAM\n', params.N, params.M);
fprintf('Iterations: %d, SNR: %d to %d dB\n', ...
    params.num_iter, params.SNR_dB(1), params.SNR_dB(end));
fprintf('Max Doppler: %.2f Hz\n', params.max_doppler);
fprintf('=====================================\n\n');

%% ================= INITIALIZE RESULTS STRUCTURE ========
results = struct();

% OCDM results
results.ocdm.ber_lmmse = zeros(size(params.SNR_dB));
results.ocdm.ber_mmsesd = zeros(size(params.SNR_dB));
results.ocdm.name = 'OCDM';

% AFDM results
results.afdm.ber_lmmse = zeros(size(params.SNR_dB));
results.afdm.ber_mmsesd = zeros(size(params.SNR_dB));
results.afdm.name = 'AFDM';

% OTFS results
results.otfs.ber_lmmse = zeros(size(params.SNR_dB));
results.otfs.ber_mmsesd = zeros(size(params.SNR_dB));
results.otfs.name = 'OTFS';

%% ================= SYSTEM 1: OCDM =================
fprintf('\n========== Running OCDM Simulation ==========\n');
results.ocdm = run_ocdm_simulation(params);

%% ================= SYSTEM 2: AFDM =================
fprintf('\n========== Running AFDM Simulation ==========\n');
results.afdm = run_afdm_simulation(params);

%% ================= SYSTEM 3: OTFS =================
fprintf('\n======= Running OTFS Simulation =====\n');
results.otfs = run_otfs_simulation(params);

%% ================= COMPARATIVE PLOTTING ========
plot_comparison_results(results, params);

%% ================= SAVE RESULTS ================
save('unified_comparison_results_NTND.mat', 'results', 'params');
fprintf('\nResults saved to: unified_comparison_results.mat\n');

%% ================= PERFORMANCE SUMMARY =================
print_performance_summary(results, params);

%% ================================================
%% ================= SYSTEM FUNCTIONS =============
%% ================================================

function results = run_ocdm_simulation(params)
    % OCDM with Fresnel Transform
    
    results.name = 'OCDM';
    results.ber_lmmse = zeros(size(params.SNR_dB));
    results.ber_mmsesd = zeros(size(params.SNR_dB));
    
    N = params.N;
    
    % Fresnel Transform Matrices
    fprintf('  Computing Fresnel matrices...\n');
    FSnT = zeros(N, N);
    for k = 1:N
        for l = 1:N
            FSnT(k,l) = 1/sqrt(N) * exp(-1j*pi/4) * exp(1j*pi*(k-l)^2/N);
        end
    end
    IFSnT = FSnT';
    
    % Cyclic Prefix
    cp_len = N/4;
    I = eye(N);
    CP_mtx = [I(N-cp_len+1:N,:); I];
    R_mtx = [zeros(N, cp_len), eye(N)];
    
    % SNR Loop
    for idx = 1:length(params.SNR_dB)
        N0 = 1/params.SNR(idx);
        err_lmmse = 0;
        err_mmsesd = 0;
        tot_bits = 0;
        
        fprintf('  SNR = %d dB: ', params.SNR_dB(idx));
        
        for it = 1:params.num_iter
            if mod(it, 100) == 0
                fprintf('.');
            end
            
            % Transmitter
            x = randi([0 params.M-1], N, 1);
            bits_tx = de2bi(x, params.bits_in_sym, 'left-msb');
            s_qam = qammod(x, params.M, 'UnitAveragePower', true);
            s_ocdm = IFSnT * s_qam;
            s_cp = CP_mtx * s_ocdm;
            
            % Channel
            [H, ~, ~, ~, ~, ~] = NTN_channels2( ...
                params.K, params.L, cp_len, params.df, ...
                params.max_doppler, params.channel_type);
            
            % AWGN
            w = sqrt(N0/2) * (randn(size(s_cp)) + 1j*randn(size(s_cp)));
            r_cp = H*s_cp + w;
            
            % Receiver
            r = R_mtx * r_cp;
            r_ocdm = FSnT * r;
            
            % Effective channel
            Heff = R_mtx * H * CP_mtx;
            D = FSnT * Heff * IFSnT;
            D = D / sqrt(trace(D*D')/N);
            
            % LMMSE
            W = D' / (D*D' + N0*eye(N));
            s_hat_lmmse = W * r_ocdm;
            x_hat_lmmse = qamdemod(s_hat_lmmse, params.M, 'UnitAveragePower', true);
            bits_lmmse = de2bi(x_hat_lmmse, params.bits_in_sym, 'left-msb');
            
            % MMSE-SD
            s_hat_mmsesd = mmse_sd_detector_unified(r_ocdm, D, N0, params.M);
            x_hat_mmsesd = qamdemod(s_hat_mmsesd, params.M, 'UnitAveragePower', true);
            bits_mmsesd = de2bi(x_hat_mmsesd, params.bits_in_sym, 'left-msb');
            
            % Count errors
            err_lmmse = err_lmmse + sum(bits_tx(:) ~= bits_lmmse(:));
            err_mmsesd = err_mmsesd + sum(bits_tx(:) ~= bits_mmsesd(:));
            tot_bits = tot_bits + numel(bits_tx);
        end
        
        results.ber_lmmse(idx) = err_lmmse / tot_bits;
        results.ber_mmsesd(idx) = err_mmsesd / tot_bits;
        
        fprintf(' LMMSE=%.3e, MMSE-SD=%.3e\n', ...
            results.ber_lmmse(idx), results.ber_mmsesd(idx));
    end
end

function results = run_afdm_simulation(params)
    % AFDM with Discrete Affine Fourier Transform
    
    results.name = 'AFDM';
    results.ber_lmmse = zeros(size(params.SNR_dB));
    results.ber_mmsesd = zeros(size(params.SNR_dB));
    
    N = params.N;
    
    % AFDM Parameters
    nu_max = params.max_doppler / params.df;
    c1 = (2*(nu_max + 2)+1)/(2*N);
    c2 = 1/(2*pi*N);
    
    fprintf('  Computing DAFT matrices...\n');
    n = (0:N-1).';
    m = (0:N-1).';
    D1 = diag(exp(1j*2*pi*c1*n.^2));
    D2 = diag(exp(1j*2*pi*c2*m.^2));
    DFT = dftmtx(N)/sqrt(N);
    A = D2 * DFT * D1;      % DAFT matrix
    AH = A';                 % Inverse DAFT
    
    % SNR Loop
    for idx = 1:length(params.SNR_dB)
        N0 = 1/params.SNR(idx);
        err_lmmse = 0;
        err_mmsesd = 0;
        tot_bits = 0;
        
        fprintf('  SNR = %d dB: ', params.SNR_dB(idx));
        
        for it = 1:params.num_iter
            if mod(it, 100) == 0
                fprintf('.');
            end
            
            % Transmitter
            x = randi([0 params.M-1], N, 1);
            bits_tx = de2bi(x, params.bits_in_sym, 'left-msb');
            y = qammod(x, params.M, 'UnitAveragePower', true);
            s = AH * y;
            s = s / sqrt(mean(abs(s).^2));
            
            % Channel
            [HT, ~, ~, ~, ~, ~] = NTN_channels1( ...
                params.K, params.L, params.df, ...
                params.max_doppler, c1, params.channel_type);
            HT = HT / sqrt(trace(HT*HT')/N); % normalization of channel

            
            % Add noise in time domain
            w_time = sqrt(N0/2) * (randn(N,1) + 1j*randn(N,1));
            r_time = HT * s + w_time;
            
            % Affine domain
            Y = A * r_time;
            H_daft = A * HT * AH;
            H_daft = H_daft / sqrt(trace(H_daft*H_daft')/N);
            
            % LMMSE
            W = H_daft' / (H_daft*H_daft' + N0*eye(N));
            y_hat_lmmse = W * Y;
            x_hat_lmmse = qamdemod(y_hat_lmmse, params.M, 'UnitAveragePower', true);
            bits_lmmse = de2bi(x_hat_lmmse, params.bits_in_sym, 'left-msb');
            
            % MMSE-SD
            y_hat_mmsesd = mmse_sd_detector_unified(Y, H_daft, N0, params.M);
            x_hat_mmsesd = qamdemod(y_hat_mmsesd, params.M, 'UnitAveragePower', true);
            bits_mmsesd = de2bi(x_hat_mmsesd, params.bits_in_sym, 'left-msb');
            
            % Count errors
            err_lmmse = err_lmmse + sum(bits_tx(:) ~= bits_lmmse(:));
            err_mmsesd = err_mmsesd + sum(bits_tx(:) ~= bits_mmsesd(:));
            tot_bits = tot_bits + numel(bits_tx);
        end
        
        results.ber_lmmse(idx) = err_lmmse / tot_bits;
        results.ber_mmsesd(idx) = err_mmsesd / tot_bits;
        
        fprintf(' LMMSE=%.3e, MMSE-SD=%.3e\n', ...
            results.ber_lmmse(idx), results.ber_mmsesd(idx));
    end
end

function results = run_otfs_simulation(params)
    % OTFS with Delay-Doppler domain processing
    
    results.name = 'OTFS';
    results.ber_lmmse = zeros(size(params.SNR_dB));
    results.ber_mmsesd = zeros(size(params.SNR_dB));
    
    M = params.K;  % Time slots
    N = params.L;  % Subcarriers
    S = M * N;     % Total symbols
    
    % OTFS Operators
    fprintf('  Computing OTFS operators...\n');
    W = dftmtx(N)/sqrt(N);
    WH = W';
    IL = eye(M);
    OP = kron(WH, IL);
    
    % SNR Loop
    for idx = 1:length(params.SNR_dB)
        N0 = 1/params.SNR(idx);
        err_lmmse = 0;
        err_mmsesd = 0;
        tot_bits = 0;
        
        fprintf('  SNR = %d dB: ', params.SNR_dB(idx));
        
        for it = 1:params.num_iter
            if mod(it, 100) == 0
                fprintf('.');
            end
            
            % Transmitter
            data = randi([0 params.M-1], S, 1);
            bits_tx = de2bi(data, params.bits_in_sym, 'left-msb');
            x = qammod(data, params.M, 'UnitAveragePower', true);
            
            % Channel
            [HT, ~, ~, ~, ~, ~] = NTN_channels( ...
                M, N, params.df, params.max_doppler, params.channel_type);
            
            % Effective DD channel
            H_eff = OP' * HT * OP;
            H_eff = H_eff / sqrt(trace(H_eff*H_eff')/S);
            
            % DD-domain signal
            w = sqrt(N0/2) * (randn(S,1) + 1j*randn(S,1));
            r_dd = H_eff*x + w;
            
            % LMMSE
            W_lmmse = (H_eff'*H_eff + N0*eye(S)) \ H_eff';
            x_lmmse = W_lmmse * r_dd;
            data_lmmse = qamdemod(x_lmmse, params.M, 'UnitAveragePower', true);
            bits_lmmse = de2bi(data_lmmse, params.bits_in_sym, 'left-msb');
            
            % MMSE-SD
            x_mmsesd = mmse_sd_detector_unified(r_dd, H_eff, N0,params.M);
            data_mmsesd = qamdemod(x_mmsesd, params.M, 'UnitAveragePower', true);
            bits_mmsesd = de2bi(data_mmsesd, params.bits_in_sym, 'left-msb');
            
            % Count errors
            err_lmmse = err_lmmse + sum(bits_tx(:) ~= bits_lmmse(:));
            err_mmsesd = err_mmsesd + sum(bits_tx(:) ~= bits_mmsesd(:));
            tot_bits = tot_bits + numel(bits_tx);
        end
        
        results.ber_lmmse(idx) = err_lmmse / tot_bits;
        results.ber_mmsesd(idx) = err_mmsesd / tot_bits;
        
        fprintf(' LMMSE=%.3e, MMSE-SD=%.3e\n', ...
            results.ber_lmmse(idx), results.ber_mmsesd(idx));
    end
end

function plot_comparison_results(results, params)
    % Create comprehensive comparison plots
    
    figure;
    
    % Subplot 1: LMMSE Comparison
    subplot(1,2,1);
    semilogy(params.SNR_dB, results.ocdm.ber_lmmse, '-r', 'LineWidth', 2);
    hold on;
    semilogy(params.SNR_dB, results.afdm.ber_lmmse, '-^g', 'LineWidth', 2);
    semilogy(params.SNR_dB, results.otfs.ber_lmmse, '-sb', 'LineWidth', 2);
    grid on;
    xlabel('SNR (dB)', 'FontName', 'Times New Roman', 'FontSize', 14);
    ylabel('BER','FontName','TimesNewRoman',  'FontSize', 14);
    title('LMMSE Detector Comparison', 'FontName','TimesNewRoman',  'FontSize', 14);
    legend('OCDM', 'AFDM', 'OTFS', 'Location', 'southwest');
    ylim([1e-7 1]);
    
    % Subplot 2: MMSE-SD Comparison
    subplot(1,2,2);
    semilogy(params.SNR_dB, results.ocdm.ber_mmsesd, '-r', 'LineWidth', 2 );
    hold on;
    semilogy(params.SNR_dB, results.afdm.ber_mmsesd, '-g', 'LineWidth', 2);
    semilogy(params.SNR_dB, results.otfs.ber_mmsesd, '-b', 'LineWidth', 2);
    grid on;
    xlabel('SNR (dB)', 'FontName','TimesNewRoman',  'FontSize', 14);
    ylabel('BER', 'FontName','TimesNewRoman',  'FontSize', 14);
    title('MMSE-SD Detector Comparison', 'FontName','TimesNewRoman',  'FontSize', 14);
    legend('OCDM', 'AFDM', 'OTFS', 'Location', 'southwest');
    ylim([1e-7 1]);
    

    saveas(gcf, 'unified_comparison.fig');
    saveas(gcf, 'unified_comparison.png');
 end

function print_performance_summary(results, params)
    % Print comprehensive performance summary
    
    fprintf('\n========================================\n');
    fprintf('      PERFORMANCE SUMMARY\n');
    fprintf('========================================\n\n');
    
    systems = {'ocdm', 'afdm', 'otfs'};
    names = {'OCDM', 'AFDM', 'OTFS'};
    
    % Find best system at each SNR
    for idx = 1:length(params.SNR_dB)
        fprintf('SNR = %d dB:\n', params.SNR_dB(idx));
        fprintf('  System    | LMMSE BER  | MMSE-SD BER | SD Gain (dB)\n');
        fprintf('  -------------------------------------------------------\n');
        
        for s = 1:length(systems)
            sys = systems{s};
            ber_l = results.(sys).ber_lmmse(idx);
            ber_s = results.(sys).ber_mmsesd(idx);
            gain = 10*log10(ber_l / ber_s);
            
            fprintf('  %-8s  | %.4e | %.4e  | %+6.2f\n', ...
                names{s}, ber_l, ber_s, gain);
        end
        fprintf('\n');
    end
    
    % Overall best performance
    fprintf('========================================\n');
    fprintf('BEST PERFORMING SYSTEM:\n');
    fprintf('========================================\n');
    
    for idx = 1:length(params.SNR_dB)
        % LMMSE
        bers_lmmse = [results.ocdm.ber_lmmse(idx), ...
                      results.afdm.ber_lmmse(idx), ...
                      results.otfs.ber_lmmse(idx)];
        [~, best_l] = min(bers_lmmse);
        
        % MMSE-SD
        bers_mmsesd = [results.ocdm.ber_mmsesd(idx), ...
                       results.afdm.ber_mmsesd(idx), ...
                       results.otfs.ber_mmsesd(idx)];
        [~, best_s] = min(bers_mmsesd);
        
        fprintf('SNR %2d dB: LMMSE → %s, MMSE-SD → %s\n', ...
            params.SNR_dB(idx), names{best_l}, names{best_s});
    end
    
    fprintf('========================================\n\n');

end
