clear all;close all;clc
%% config
addpath '..\Functions'

% Parameters of the KS equation
KS_N = 32;
dim = KS_N;
lyapunov_exp = 520; %%
KS_L = 1/2;
KS_v = 4;
solver_tstep = 5e-6;

reservoir_tstep = 2e-5;

train_r_step_cut = ceil( 6 / reservoir_tstep ); % drop the transient in data
train_r_step_length = 520000;
validate_r_step_length = floor( 15 / lyapunov_exp /reservoir_tstep );

para_train_set = [196 197 198];
tp_train_set = para_train_set;


success_threshold = 8;


KS_tmax = ( train_r_step_cut + train_r_step_length + validate_r_step_length + 100 ) * reservoir_tstep; % time, for timeseries
rng('shuffle');
tic;

%% preparing training data
fprintf('preparing training data...\n');
train_data = zeros(length(tp_train_set), train_r_step_length + validate_r_step_length + 8 ,dim+1); % data that goes into reservior_training
for tp_train_i = 1:length(tp_train_set)
    tp = tp_train_set(tp_train_i);
    KS_alpha = para_train_set(tp_train_i);  %% system sensitive    
    [Adata,X,~] = func_KSsim1D(KS_alpha,KS_N,KS_L,KS_v,KS_tmax,reservoir_tstep,solver_tstep);
    toc;
    
    Adata = Adata(train_r_step_cut+1:end,:);
    train_data(tp_train_i,:,1:KS_N) = Adata(1:size(train_data,2),:);    
end


%% preparing testing data

para_test = 200.14;
test_r_step_length = 50000;
fprintf('preparing testing data...\n');

KS_alpha = para_test;  %% system sensitive    
[Adata,X,~] = func_KSsim1D(KS_alpha,KS_N,KS_L,KS_v,KS_tmax,reservoir_tstep,solver_tstep);
toc;

Adata = Adata(train_r_step_cut+1:end,:);
test_data = Adata;


% save the training and testing data with mat format
save('KS_train_data.mat', 'train_data');
save('KS_test_data.mat', 'test_data');
