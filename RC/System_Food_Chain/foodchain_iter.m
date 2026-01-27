
%% -------------------- Paths --------------------
addpath('..\Functions');  % adjust if needed
save_dir = './save_for_plot';
if ~exist(save_dir,'dir'), mkdir(save_dir); end

%% -------------------- System params  --------------------
xc = 0.4;  yc = 2.009;  xp = 0.08;  yp = 2.876;  r0 = 0.16129;  c0 = 0.5;
dim = 3;

%% -------------------- Reservoir hyperparameters  --------------------
n = 900;          % hidden size
k_avg = 4;        % avg degree 
eig_rho = 2.3;    % spectral radius of W_r
W_in_a = 3.6;     % input scaling
tp_W = 0.50;     % scaling coefficient before bifurcation parameter
tp_bias = -2.2;  % bias coefficient added to bifurcation parameter
tp_bias = 0;
a = 0.30;        % leakage
beta = 3e-5;     % L2 regularization for readout

reservoir_tstep = 1;   % RC internal step
ratio_tstep = 10;      % RK4 step = reservoir_tstep/ratio_tstep

% Training data lengths (per k)
train_r_step_cut       = 50 * 40 / reservoir_tstep;   % drop transient
train_r_step_length    = 80 * 40 / reservoir_tstep;   % training
validate_r_step_length = 30 * 40 / reservoir_tstep;   % validation

% Param set for training (match Transformer):
para_train_set = [0.97 0.98 0.99];   % k values for training

%% -------------------- Iteration & scan settings --------------------
num_models     = 1000;      % train 50 independent RC models
runs_per_model = 1;      % 20 scans per model

sequence_length = 512;    % warmup context length (match Transformer)
rollout_steps   = 2000;   % prediction length per scan

k_min = 0.995; k_max = 1.005; k_step = 0.0001;   % sweep grid
k_values = k_min:k_step:k_max;

z_threshold = 0.4;        % collapse threshold on P (3rd state)

%% -------------------- Helper: choose random warmup start --------------------
choose_start = @(T,L) randi([1, max(1, T - L + 1)]);

%% -------------------- Precompute ODE training trajectories --------------------
% Build training data for func_STP_train: per-k time series + parameter channel
fprintf('Preparing RC training data...\n');
train_data_length = train_r_step_length + validate_r_step_length + 10;
train_data = zeros(length(para_train_set), train_data_length, dim+1);

for tp_i = 1:length(para_train_set)
    k_train = para_train_set(tp_i);
    flag_ghost = [xc yc xp yp r0 c0 k_train];

    % Generate a sufficiently long accurate trajectory via RK4
    tmax_timeseries_train = (train_r_step_cut + train_r_step_length + validate_r_step_length + 20) * reservoir_tstep;
    ts_train = zeros(1,dim);
    % ensure predator (3rd state) not too small in training window
    while min(ts_train(:,3)) < 0.5
        x0 = [ 0.6 + 0.4*rand; 0.15 + 0.4*rand; 0.3 + 0.5*rand ];
        [~, ts_full] = ode4(@(t,x) eq_ghost(t,x,flag_ghost), 0:reservoir_tstep/ratio_tstep:tmax_timeseries_train, x0);
        ts_train = ts_full(1:ratio_tstep:end,:);      % downsample to reservoir_tstep
        ts_train = ts_train(train_r_step_cut+1:end,:);% drop transient
    end

    % Assemble train_data: states + parameter channel
    train_data(tp_i,1:train_data_length,1:dim)   = ts_train(1:train_data_length,:);
    train_data(tp_i,1:train_data_length,dim+1)   = tp_W * (k_train + tp_bias);
end

%% -------------------- Generate ONE long warmup trajectory at k=0.99 --------------------
k_warmup = 0.99;
flag_ghost_warm = [xc yc xp yp r0 c0 k_warmup];

warmup_r_step_cut    = 4000 - sequence_length;  % drop transient
warmup_pool_extra    = 20000;                   % extra samples to allow many random windows
warmup_total_len     = warmup_r_step_cut + sequence_length + warmup_pool_extra;

fprintf('Generating warmup trajectory at k=0.99...\n');
ts_warm = zeros(1,dim);
while min(ts_warm(:,3)) < 0.5
    x0 = [ 0.6 + 0.4*rand; 0.15 + 0.4*rand; 0.3 + 0.5*rand ];
    [~, ts_full_w] = ode4(@(t,x) eq_ghost(t,x,flag_ghost_warm), 0:reservoir_tstep/ratio_tstep:warmup_total_len, x0);
    ts_warm = ts_full_w(1:ratio_tstep:end,:);                   % downsample
    ts_warm = ts_warm(warmup_r_step_cut+1:end,:);               % drop transient
end
T_warm = size(ts_warm,1);

%% -------------------- Train+Scan loops --------------------
all_critical_ks = [];
records = struct('model_index',{},'run_index',{},'start',{},'critical_k',{},'critical_idx',{});

for mi = 1:num_models
    fprintf('\n[RC] Training model %02d/%02d...\n', mi, num_models);
    rng(1234 + mi);  % per-model seed for reproducibility

    % Train once (no best-of) to mirror Transformer fairness
    flag_r_train = [n k_avg eig_rho W_in_a a beta train_r_step_cut train_r_step_length ...
                    validate_r_step_length reservoir_tstep dim];
    [rmse,W_in,W_r,W_out,~,~,~] = func_STP_train(train_data, para_train_set, flag_r_train, 1,1,1);
    % fprintf('[RC %02d] train RMSE = %.6f\n', mi, rmse);

    % Per-model scans
    rng(777 + mi);
    for ri = 1:runs_per_model
        % Random warmup window from the generated k=0.99 trajectory
        start_idx = choose_start(T_warm, sequence_length);
        x_warmup  = ts_warm(start_idx : start_idx + sequence_length - 1, :);  % [L x 3]

        % Predict for increasing k; find first critical k
        critical_k = NaN; critical_idx = NaN;
        for kv = 1:numel(k_values)
            k_now = k_values(kv);
            tp_now = k_now;  % reservoir parameter 

            % RC prediction
            warmup_len = size(x_warmup,1);
            predict_r_step_cut = 0;
            predict_r_step_length = rollout_steps;
            flag_r = [n dim a warmup_len predict_r_step_cut predict_r_step_length];
            predict_r = func_STP_predict(x_warmup, tp_W * (tp_now + tp_bias), W_in, W_r, W_out, flag_r);

            % Detect collapse by threshold on P (3rd state)
            z = predict_r(:,3);
            below = find(z < z_threshold, 1, 'first');
            if ~isempty(below)
                critical_k = k_now;
                critical_idx = below; 
                break;
            end
        end

        % Log
        records(end+1).model_index = mi;
        records(end).run_index  = ri;
        records(end).start      = start_idx;
        if ~isnan(critical_k)
            records(end).critical_k  = critical_k;
            records(end).critical_idx= critical_idx;
            all_critical_ks(end+1)   = critical_k; 
        else
            records(end).critical_k  = NaN;
            records(end).critical_idx= NaN;
        end

        fprintf('[RC %02d] run %02d: critical k = %s\n', mi, ri, mat2str(critical_k,6));
    end
end

%% -------------------- Save + Plot --------------------
S = struct();
S.config = struct('num_models',num_models,'runs_per_model',runs_per_model, ...
    'sequence_length',sequence_length,'rollout_steps',rollout_steps, ...
    'k_min',k_min,'k_max',k_max,'k_step',k_step,'z_threshold',z_threshold, ...
    'k_warmup', k_warmup);
S.records = records;
S.critical_k_values = all_critical_ks;

% save(fullfile(save_dir,'rc_foodchain_critical_points.mat'),'S');
% fprintf('[save] rc results -> %s (N=%d)\n', fullfile(save_dir,'rc_foodchain_critical_points.mat'), numel(all_critical_ks));

% Histogram (bins aligned to k grid)
valid = all_critical_ks(~isnan(all_critical_ks));
figure('Color','w');
if ~isempty(valid)
    edges = k_min:k_step:(k_max + k_step);
    histogram(valid, edges);
    xlabel('Predicted critical k'); ylabel('Frequency');
    title('RC: Distribution of predicted critical points (foodchain, k=0.99 warmup)');
else
    warning('No critical points detected; histogram skipped.');
end
