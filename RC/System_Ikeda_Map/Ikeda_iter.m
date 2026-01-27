
%% -------------------- Config --------------------
addpath('..\\Functions');
save_dir = './save_for_plot';
if ~exist(save_dir,'dir'), mkdir(save_dir); end

dim = 2;   % Ikeda map is 2D

% Reservoir hyperparameters
n = 400;
k_avg = 283;        
eig_rho = 0.17;     
W_in_a = 2.6;       
tp_W = 0.35;       
tp_bias = 0.47;    
a = 1.0;           
beta = 1e-6;       

reservoir_tstep = 1;  
ratio_tstep = 1;     

train_r_step_cut        = round( 20 / reservoir_tstep );
train_r_step_length     = round( 800 / reservoir_tstep );
validate_r_step_length  = round(  15 / reservoir_tstep );

para_train_set = [0.91 0.94 0.97];

%% -------------------- Iteration & scan settings --------------------
num_models     = 50;
runs_per_model = 20;

sequence_length = 512;
rollout_steps   = 2000;

mu_min  = 0.97;
mu_max  = 1.05;
mu_step = 0.001;
mu_values = mu_min:mu_step:mu_max;

% Collapse bounds
x_min = -1.0; x_max =  2.2;   
y_min = -2.5; y_max =  2.0;   

%% -------------------- Build training data --------------------
fprintf('Preparing RC training data...\n');
train_data_length = train_r_step_length + validate_r_step_length + 10;
train_data = zeros(length(para_train_set), train_data_length, dim+1);

for miu = 1:length(para_train_set)
    mu_train = para_train_set(miu);
    total_len = (train_r_step_cut + train_r_step_length + validate_r_step_length + 20);

    % Generate Ikeda trajectory inline
    z = (rand + 1i*rand);
    ts_full = zeros(total_len, 2);
    for t = 1:total_len
        z = func_IHJM(z, mu_train);
        ts_full(t,1) = real(z);
        ts_full(t,2) = imag(z);
    end

    ts_train = ts_full(train_r_step_cut+1:end, :);  % drop transient

    train_data(miu,1:train_data_length,1:dim) = ts_train(1:train_data_length,:);
    train_data(miu,1:train_data_length,dim+1) = tp_W * (mu_train + tp_bias);
end

%% -------------------- Warmup trajectory at mu_warmup --------------------
mu_warmup = max(para_train_set);  % 0.97
warmup_cut   = 4000 - sequence_length;
warmup_extra = 20000;
warmup_total = warmup_cut + sequence_length + warmup_extra;

fprintf('Generating warmup trajectory at mu = %.3f...\n', mu_warmup);
z = (rand + 1i*rand);
ts_warm_full = zeros(warmup_total, 2);
for t = 1:warmup_total
    z = func_IHJM(z, mu_warmup);
    ts_warm_full(t,1) = real(z);
    ts_warm_full(t,2) = imag(z);
end
ts_warm = ts_warm_full(warmup_cut+1:end, :);
T_warm = size(ts_warm,1);

%% -------------------- Train + Scan loops --------------------
all_critical_mu = [];
records = struct('model_index',{},'run_index',{},'start',{},'critical_mu',{},'critical_idx',{});

for mi = 1:num_models
    fprintf('\n[RC] Training model %02d/%02d...\n', mi, num_models);
    rng(1234 + mi);

    flag_r_train = [n k_avg eig_rho W_in_a a beta train_r_step_cut train_r_step_length ...
                    validate_r_step_length reservoir_tstep dim];
    [rmse,W_in,W_r,W_out,~,~,~] = func_STP_train(train_data, para_train_set, flag_r_train, 1,1,1);
    fprintf('[RC %02d] train RMSE = %.6f\n', mi, rmse);

    rng(777 + mi);
    for ri = 1:runs_per_model
        start_idx = randi([1, max(1, T_warm - sequence_length + 1)]);
        x_warmup  = ts_warm(start_idx : start_idx + sequence_length - 1, :);

        critical_mu = NaN; critical_idx = NaN;
        for mv = 1:numel(mu_values)
            mu_now = mu_values(mv);

            warmup_len = size(x_warmup,1);
            predict_r_step_cut = 0;
            predict_r_step_length = rollout_steps;
            flag_r = [n dim a warmup_len predict_r_step_cut predict_r_step_length];
            predict_r = func_STP_predict(x_warmup, tp_W * (mu_now + tp_bias), W_in, W_r, W_out, flag_r);

            x = predict_r(:,1); y = predict_r(:,2);
            idx_x = find(x < x_min | x > x_max, 1, 'first');
            idx_y = find(y < y_min | y > y_max, 1, 'first');

            if ~isempty(idx_x) || ~isempty(idx_y)
                if isempty(idx_x), idx_x = inf; end
                if isempty(idx_y), idx_y = inf; end
                critical_mu = mu_now;
                critical_idx = min(idx_x, idx_y);
                break;
            end
        end

        records(end+1).model_index = mi; 
        records(end).run_index = ri;
        records(end).start = start_idx;
        if ~isnan(critical_mu)
            records(end).critical_mu = critical_mu;
            records(end).critical_idx = critical_idx;
            all_critical_mu(end+1) = critical_mu; 
        else
            records(end).critical_mu = NaN;
            records(end).critical_idx = NaN;
        end
        fprintf('[RC %02d] run %02d: critical mu = %s\n', mi, ri, mat2str(critical_mu,6));
    end
end

%% -------------------- Save + Plot --------------------
S = struct();
S.config = struct('num_models',num_models,'runs_per_model',runs_per_model, ...
    'sequence_length',sequence_length,'rollout_steps',rollout_steps, ...
    'mu_min',mu_min,'mu_max',mu_max,'mu_step',mu_step, ...
    'x_min',x_min,'x_max',x_max,'y_min',y_min,'y_max',y_max, ...
    'mu_warmup', mu_warmup);
S.records = records;
S.critical_mu_values = all_critical_mu;

save(fullfile(save_dir,'rc_ikeda_critical_points.mat'),'S');
fprintf('[save] rc results -> %s (N=%d)\n', fullfile(save_dir,'rc_ikeda_critical_points.mat'), numel(all_critical_mu));

valid = all_critical_mu(~isnan(all_critical_mu));
figure('Color','w');
if ~isempty(valid)
    edges = mu_min:mu_step:(mu_max + mu_step);
    histogram(valid, edges);
    xlabel('Predicted critical \\mu'); ylabel('Frequency');
    title('RC: Distribution of predicted critical points (Ikeda, warmup at max train \\mu)');
else
    warning('No critical points detected; histogram skipped.');
end

