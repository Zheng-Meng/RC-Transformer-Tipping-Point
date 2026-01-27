clear all;close all; clc
%% -------------------- Paths --------------------
addpath('..\Functions');  % adjust if needed
save_dir = './save_for_plot';
if ~exist(save_dir,'dir'), mkdir(save_dir); end

%% config
addpath '..\Functions'

% Parameters of the KS equation
KS_N = 32;
dim = KS_N;
lyapunov_exp = 520; %%
KS_L = 1/2;
KS_v = 4;
solver_tstep = 5e-6;


% Hyperparameters of the reservoir
n = 1000; % Maybe it does not have to be so large
k = 450;
eig_rho = 0.89;
W_in_a = 0.057;
tp_W = -0.052;
tp_bias = -185;
a = 1;
beta = 8 * 10^(-5);


reservoir_tstep = 2e-5;
dt = reservoir_tstep;

train_r_step_cut = ceil( 6 / reservoir_tstep ); % drop the transient in data
train_r_step_length = 12000;
validate_r_step_length = floor( 15 / lyapunov_exp /reservoir_tstep );

para_train_set = [196 197 198];
tp_train_set = para_train_set;


success_threshold = 8;


KS_tmax = ( train_r_step_cut + train_r_step_length + validate_r_step_length + 100 ) * reservoir_tstep; % time, for timeseries
rng('shuffle');
tic;


%% -------------------- Iteration & scan settings --------------------
num_models     = 50;      % train 50 independent RC models
runs_per_model = 1;      % 20 scans per model

sequence_length = 512;    % warmup context length (match Transformer)
% rollout_steps   =  floor( 0.5 / reservoir_tstep );   % prediction length per scan
rollout_steps = 20000;

alpha_min  = 199.7;
alpha_max  = 200.8;
alpha_step = 0.03;
alpha_values = alpha_min:alpha_step:alpha_max;

% Choose one alpha for warmup trajectory
alpha_warmup = 198;       % similar idea as k=0.99 warmup for foodchain
fprintf('Warmup alpha (KS_alpha) = %.3f\n', alpha_warmup);

%% --------- Generate a long warmup trajectory at alpha_warmup ---------
warmup_r_step_cut = ceil( 6 / reservoir_tstep );  % drop transient (similar style as training)
warmup_pool_extra = 20000;                        % extra length to allow many random windows
warmup_total_len  = warmup_r_step_cut + sequence_length + warmup_pool_extra;

fprintf('Generating warmup trajectory for KS alpha = %.3f...\n', alpha_warmup);
ts_warm = zeros(1, dim);
while true
    % random initial condition in some range 
    x0 = rand(1, dim) * 0.1;  % small random perturbation
    [Adata_warm, X_ks, ~] = func_KSsim1D(alpha_warmup, KS_N, KS_L, KS_v, ...
        warmup_total_len * reservoir_tstep, reservoir_tstep, solver_tstep);
    % Adata_warm: [T_full x KS_N]
    % No extra downsampling needed: func_KSsim1D already uses reservoir_tstep.
    ts_warm = Adata_warm(warmup_r_step_cut+1:end, :); % drop transient
    if size(ts_warm,1) >= sequence_length + 1000
        break;
    end
end
T_warm = size(ts_warm, 1);

% Helper to pick a random warmup start
choose_start = @(T,L) randi([1, max(1, T-L+1)]);

%% -------------------- Train+Scan loops --------------------
all_critical_ks = [];
records = struct('model_index',{},'run_index',{},'start',{},'critical_k',{});

for mi = 1:num_models
    fprintf('\n[RC] Training model %02d/%02d...\n', mi, num_models);
    rng(1234 + mi);  % per-model seed for reproducibility

    fprintf('preparing training data...\n');
    train_data = zeros(length(tp_train_set), train_r_step_length + validate_r_step_length + 8 ,dim+1); % data that goes into reservior_training
    for tp_train_i = 1:length(tp_train_set)
        tp = tp_train_set(tp_train_i);
        KS_alpha = para_train_set(tp_train_i);  %% system sensitive    
        [Adata,X,~] = func_KSsim1D(KS_alpha,KS_N,KS_L,KS_v,KS_tmax,reservoir_tstep,solver_tstep);
        toc;
        
        Adata = Adata(train_r_step_cut+1:end,:);
        train_data(tp_train_i,:,1:KS_N) = Adata(1:size(train_data,2),:);    
        train_data(tp_train_i,:,dim+1) = tp_W * (tp + tp_bias) * ones(size(train_data,2),1);    %% system sensitive
    end

    flag_r_train = [n k eig_rho W_in_a a beta train_r_step_cut train_r_step_length validate_r_step_length...
        reservoir_tstep dim success_threshold];
    [success_length,W_in,W_r,W_out,t_validate_temp,x_real_temp,x_validate_temp] = ...
        func_STP_train(train_data,tp_train_set,flag_r_train,2,2,2);
    
    disp(['training success length:', num2str(success_length * lyapunov_exp)])
    % Per-model scans
    rng(777 + mi);
    for ri = 1:runs_per_model
        % Random warmup window from the generated k=0.99 trajectory
        start_idx = choose_start(T_warm, sequence_length);
        x_warmup  = ts_warm(start_idx : start_idx + sequence_length - 1, :);  % [L x 3]

        % Predict for increasing k; find first critical k
        critical_k = NaN; 
        for alpha_id = 1:numel(alpha_values)
            alpha_now = alpha_values(alpha_id);
            tp_now = alpha_now;  % reservoir parameter 

            % RC prediction
            warmup_len = size(x_warmup,1);
            predict_r_step_cut = 0;
            predict_r_step_length = rollout_steps;
            flag_r = [n dim a warmup_len predict_r_step_cut predict_r_step_length];
            predict_r = func_STP_predict(x_warmup, tp_W * (tp_now + tp_bias), W_in, W_r, W_out, flag_r);

            % Detect collapse by threshold on P (3rd state)
            [flag, ~, ~, ~, ~] = detect_KS_collapse_freq(predict_r, dt);

            if flag
                critical_k = alpha_now;
                break;
            end
        end

        % Log
        records(end+1).model_index = mi;
        records(end).run_index  = ri;
        records(end).start      = start_idx;
        if ~isnan(critical_k)
            records(end).critical_k  = critical_k;
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
    'alpha_min',alpha_min,'alpha_max',alpha_max,'alpha_step',alpha_step, ...
    'alpha_warmup', alpha_warmup);
S.records = records;
S.critical_k_values = all_critical_ks;

save(fullfile(save_dir,'rc_KS_critical_points.mat'),'S');
fprintf('[save] rc results -> %s (N=%d)\n', fullfile(save_dir,'rc_KS_critical_points.mat'), numel(all_critical_ks));

% Histogram (bins aligned to k grid)
% valid = all_critical_ks(~isnan(all_critical_ks));
% figure('Color','w');
% if ~isempty(valid)
%     edges = alpha_min:alpha_step:(alpha_max + alpha_step);
%     histogram(valid, edges);
%     xlabel('Predicted critical k'); ylabel('Frequency');
%     title('RC: Distribution of predicted critical points (foodchain, k=0.99 warmup)');
% else
%     warning('No critical points detected; histogram skipped.');
% end

% calculate the average critial point k, if not NaN
average_critical_k = mean(all_critical_ks(~isnan(all_critical_ks)));
fprintf('Average critical k = %.3f\n', average_critical_k);




