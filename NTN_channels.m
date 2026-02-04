function [HT, gi, li, ki, delays, pdp_lin] = NTN_channels( ...
            M, N, delta_f, max_doppler, channel_type)

% ================= Constants =================

S = M*N;
T = 1/delta_f;

% OTFS resolutions
del_resolution = 1/(M*delta_f);
do_resolution  = 1/(N*T);

% ================= Channel Selection =================
switch channel_type

    case 'NTN_A'
        delays = [0 1.0811 2.8416]*1e-6;
        pdp_dB = [0 -4.675 -6.482];
        K_dB = -inf;     % Rayleigh only

    case 'NTN_B'
        delays = [0 0.7249 0.7410 5.7392]*1e-6;
        pdp_dB = [0 -1.973 -4.332 -11.914];
        K_dB = -inf;

    case 'NTN_C'
        delays = [0 0 14.8124]*1e-6;
        pdp_dB = [-0.394 -10.618 -23.373];
        K_dB = 10.224;

    case 'NTN_D'
        delays = [0 0 0.5596 7.3340]*1e-6;
        pdp_dB = [-0.284 -11.991 -9.887 -16.771];
        K_dB = 11.707;

    otherwise
        error('Wrong channel selected');
end

% ================= PDP =================
pdp_lin = 10.^(pdp_dB/10);
pdp_lin = pdp_lin / sum(pdp_lin);
paths = length(pdp_lin);

% ================= Fading =================
gi = zeros(1,paths);

if isfinite(K_dB)    % Ricean LOS
    K = 10^(K_dB/10);
    gi(1) = sqrt(pdp_lin(1)) * ...
        ( sqrt(K/(K+1)) + ...
          sqrt(1/(K+1))*(randn+1j*randn)/sqrt(2) );

    for p = 2:paths
        gi(p) = sqrt(pdp_lin(p)) * ...
                (randn+1j*randn)/sqrt(2);
    end
else                 % Rayleigh only
    gi = sqrt(pdp_lin) .* ...
         (randn(1,paths)+1j*randn(1,paths))/sqrt(2);
end

% ================= Delay & Doppler =================
li = round(delays / del_resolution);



nu = max_doppler * cos(2*pi*rand(1,paths));   % fractional Doppler
ki = nu / do_resolution;


% ================= Channel Matrix =================
HT = zeros(S,S);
n_axis = (0:S-1)';

for p = 1:paths
    Dp = diag(exp(1j*2*pi*ki(p)*n_axis/S));
    Pp = circshift(eye(S), li(p), 1);
    HT = HT + gi(p) * (Dp * Pp);
end

end
