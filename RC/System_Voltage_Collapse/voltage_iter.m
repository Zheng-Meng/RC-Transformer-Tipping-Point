
%% -------------------- Config --------------------
addpath('..\Functions');
save_dir = './save_for_plot';
if ~exist(save_dir,'dir'), mkdir(save_dir); end

% System dimensionality
dim = 4;

% Reservoir hyperparameters 
n = 800;              % hidden size
k_avg = 250;          % avg degree (W_r)
eig_rho = 1.6;        % spectral radius of W_r
W_in_a = 2.1;         % input scaling
tp_W = 1.6;          % scaling before parameter channel
tp_bias = -3.1;      % bias added to parameter channel
a = 1.0;             % leakage
beta = 1e-4;          % L2 regularization for readout

reservoir_tstep = 0.05;   % RC internal step
ratio_tstep = 5;          % RK4 step = reservoir_tstep/ratio_tstep

% Training lengths (per Q1) 
train_r_step_cut        = round( 500 / reservoir_tstep );
train_r_step_length     = round( 500 / reservoir_tstep );
validate_r_step_length  = round(  25 / reservoir_tstep );

% Training parameter set 
para_train_set = [2.98968 2.98973 2.98978];

%% -------------------- Iteration & scan settings --------------------
num_models     = 50;      % train 50 RC models
runs_per_model = 20;      % 20 scans per model

sequence_length = 512;    % warmup context length (match Transformer)
rollout_steps   = 2000;   % prediction length per scan

q1_min  = 2.98978;        % sweep grid 
q1_max  = 2.98986;
q1_step = 0.000002;
q1_values = q1_min:q1_step:q1_max;

v_threshold = 0.75;        % collapse threshold on V (4th state)

%% -------------------- Helper: random warmup start --------------------
choose_start = @(T,L) randi([1, max(1, T - L + 1)]);

%% -------------------- Build training data (ODE per-Q1) --------------------
fprintf('Preparing RC training data...\n');
train_data_length = train_r_step_length + validate_r_step_length + 10;
train_data = zeros(length(para_train_set), train_data_length, dim+1);

for qi = 1:length(para_train_set)
    Q1_train = para_train_set(qi);

    % Generate long, accurate trajectory via RK4
    tmax_timeseries_train = (train_r_step_cut + train_r_step_length + validate_r_step_length + 20) * reservoir_tstep;
    ts_train = NaN;
    while (isempty(ts_train) || any(isnan(ts_train(end,:))))
        x0 = [ 0.17 + 0.13*rand ; 0.00 + 0.10*rand ; 0.05 + 0.10*rand ; 0.83 + 0.05*rand ];
        [~, ts_full] = ode4(@(t,x) eq_VoltageCollapse(t,x,Q1_train), 0:reservoir_tstep/ratio_tstep:tmax_timeseries_train, x0);
        % Downsample to reservoir step
        ts_ds = ts_full(1:ratio_tstep:end,:);
        % Drop transient
        if size(ts_ds,1) > train_r_step_cut
            ts_train = ts_ds(train_r_step_cut+1:end,:);
        else
            ts_train = NaN;
        end
    end

    % Assemble: states + parameter channel
    train_data(qi,1:train_data_length,1:dim) = ts_train(1:train_data_length,:);
    train_data(qi,1:train_data_length,dim+1) = tp_W * (Q1_train + tp_bias);
end

%% -------------------- Generate one long warmup trajectory at Q1_warmup --------------------
Q1_warmup = max(para_train_set);  % like using highest training param for context
warmup_r_step_cut = round( 500 / reservoir_tstep ) - sequence_length;  % ensure long post-cut sequence
warmup_pool_extra = 20000;  % extra samples to draw many random windows
warmup_total_len  = warmup_r_step_cut + sequence_length + warmup_pool_extra;

fprintf('Generating warmup trajectory at Q1 = %.6f...\n', Q1_warmup);
ts_warm = NaN;
while (isempty(ts_warm) || any(isnan(ts_warm(end,:))))
    x0 = [ 0.17 + 0.13*rand ; 0.00 + 0.10*rand ; 0.05 + 0.10*rand ; 0.83 + 0.05*rand ];
    [~, ts_full_w] = ode4(@(t,x) eq_VoltageCollapse(t,x,Q1_warmup), 0:reservoir_tstep/ratio_tstep:warmup_total_len, x0);
    ts_ds_w = ts_full_w(1:ratio_tstep:end,:);      % downsample
    if size(ts_ds_w,1) > warmup_r_step_cut
        ts_warm = ts_ds_w(warmup_r_step_cut+1:end,:);  % drop transient
    else
        ts_warm = NaN;
    end
end
T_warm = size(ts_warm,1);

%% -------------------- Train + Scan loops --------------------
all_critical_q1 = [];
records = struct('model_index',{},'run_index',{},'start',{},'critical_q1',{},'critical_idx',{});

for mi = 1:num_models
    fprintf('\n[RC] Training model %02d/%02d...\n', mi, num_models);
    rng(1234 + mi);  % per-model seed

    % Train once (no best-of) for fairness
    flag_r_train = [n k_avg eig_rho W_in_a a beta train_r_step_cut train_r_step_length ...
                    validate_r_step_length reservoir_tstep dim];
    [rmse,W_in,W_r,W_out,~,~,~] = func_STP_train(train_data, para_train_set, flag_r_train, 1,1,1);
    fprintf('[RC %02d] train RMSE = %.6f\n', mi, rmse);

    % Per-model scans
    rng(777 + mi);
    for ri = 1:runs_per_model
        % Random warmup window from generated Q1_warmup trajectory
        start_idx = choose_start(T_warm, sequence_length);
        x_warmup  = ts_warm(start_idx : start_idx + sequence_length - 1, :);  % [L x 4]

        % Sweep Q1 and detect first collapse
        critical_q1 = NaN; critical_idx = NaN;
        for qv = 1:numel(q1_values)
            Q1_now = q1_values(qv);
            tp_now = Q1_now; % parameter fed via tp_W, tp_bias

            warmup_len = size(x_warmup,1);
            predict_r_step_cut = 0;
            predict_r_step_length = rollout_steps;
            flag_r = [n dim a warmup_len predict_r_step_cut predict_r_step_length];
            predict_r = func_STP_predict(x_warmup, tp_W * (tp_now + tp_bias), W_in, W_r, W_out, flag_r);

            % Collapse detection: V (4th state) below threshold
            V = predict_r(:,4);
            idx = find(V < v_threshold, 1, 'first');
            if ~isempty(idx)
                critical_q1 = Q1_now;
                critical_idx = idx;
                break;
            end
        end

        % Log
        records(end+1).model_index = mi;
        records(end).run_index = ri;
        records(end).start = start_idx;
        if ~isnan(critical_q1)
            records(end).critical_q1 = critical_q1;
            records(end).critical_idx = critical_idx;
            all_critical_q1(end+1) = critical_q1; 
        else
            records(end).critical_q1 = NaN;
            records(end).critical_idx = NaN;
        end
        fprintf('[RC %02d] run %02d: critical Q1 = %s\n', mi, ri, mat2str(critical_q1,8));
    end
end

%% -------------------- Save + Plot --------------------
% S = struct();
% S.config = struct('num_models',num_models,'runs_per_model',runs_per_model, ...
%     'sequence_length',sequence_length,'rollout_steps',rollout_steps, ...
%     'q1_min',q1_min,'q1_max',q1_max,'q1_step',q1_step,'v_threshold',v_threshold, ...
%     'Q1_warmup', Q1_warmup);
% S.records = records;
% S.critical_q1_values = all_critical_q1;
% 
% save(fullfile(save_dir,'rc_voltage_critical_points.mat'),'S');
% fprintf('[save] rc results -> %s (N=%d)\n', fullfile(save_dir,'rc_voltage_critical_points.mat'), numel(all_critical_q1));
% 
% % Histogram aligned to Q1 grid
% valid = all_critical_q1(~isnan(all_critical_q1));
% figure('Color','w');
% if ~isempty(valid)
%     edges = q1_min:q1_step:(q1_max + q1_step);
%     histogram(valid, edges);
%     xlabel('Predicted critical Q1'); ylabel('Frequency');
%     title('RC: Distribution of predicted critical points (voltage, Q1 warmup)');
% else
%     warning('No critical points detected; histogram skipped.');
% end
