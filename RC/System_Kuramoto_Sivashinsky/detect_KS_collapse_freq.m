function [is_collapse, t_collapse, R_series, R_smooth, t_centers] = ...
    detect_KS_collapse_freq(U, dt, window_len, step, R_thresh, persist_win)
% Detect transition from chaotic to periodic using time–frequency analysis.
%
% U         : T x N  (time x space)
% dt        : time step
% window_len: window length in time steps (e.g. 2048)
% step      : step between windows (e.g. 200)
% R_thresh  : threshold for "periodic" (e.g. 0.6)
% persist_win: require this many consecutive periodic windows (e.g. 5)
%
% is_collapse : true if we see chaos first, then sustained periodic
% t_collapse  : estimated collapse time (NaN if none)
% R_series    : raw peak-power ratio per window
% R_smooth    : smoothed ratio
% t_centers   : time of each window center

    [T, N] = size(U);

    if nargin < 2 || isempty(dt),         dt         = 1;      end
    if nargin < 3 || isempty(window_len), window_len = 2048;   end
    if nargin < 4 || isempty(step),       step       = 200;    end
    if nargin < 5 || isempty(R_thresh),   R_thresh   = 0.5;    end
    if nargin < 6 || isempty(persist_win),persist_win= 5;      end

    % 1. Representative time series: choose middle spatial point
    y = U(:, round(N/2));
    y = y - mean(y);

    % 2. Sliding windows
    starts = 1:step:(T - window_len + 1);
    nW = numel(starts);
    R_series  = zeros(nW,1);
    t_centers = zeros(nW,1);

    for i = 1:nW
        t0 = starts(i);
        seg = y(t0 : t0 + window_len - 1);
        seg = seg - mean(seg);

        Y = fft(seg);
        P = abs(Y).^2;

        % use only positive frequencies, excluding DC
        kpos = 2:floor(window_len/2);
        Ppos = P(kpos);

        if sum(Ppos) <= 0
            R_series(i) = 0;
        else
            R_series(i) = max(Ppos) / sum(Ppos);
        end

        t_centers(i) = ((t0 + t0 + window_len - 1) / 2) * dt;
    end

    % 3. Smooth R to reduce noise
    R_smooth = movmean(R_series, 3);

    % 4. Detect sustained periodic regime (high R)
    is_periodic = (R_smooth > R_thresh);
    idx_blocks  = strfind(is_periodic.', ones(1, persist_win));

    if isempty(idx_blocks)
        is_collapse = false;
        t_collapse  = NaN;
        return;
    end

    % 5. Optional: check that we were NOT periodic at the very beginning
    first_idx = idx_blocks(1);
    % If before that, average R is clearly smaller, we call it a collapse
    mean_before = mean(R_smooth(1:max(first_idx-1,1)));
    mean_after  = mean(R_smooth(first_idx:end));

    if mean_after > R_thresh && mean_before < R_thresh
        is_collapse = true;
        t_collapse  = t_centers(first_idx);
    else
        is_collapse = false;
        t_collapse  = NaN;
    end
end
