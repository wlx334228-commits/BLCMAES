%RUNSMD_AUXILIARYONLYEXPERIMENT One-click formal auxiliary-only experiment.
% Runs SMD1--SMD12 with DU=2, DL=3, 20 paired runs and K=0,...,8. K=0 is
% the original BL-CMA-ES control; K=1,...,8 use auxiliary-only sampling.
% Each
% prediction-guided candidate is sampled around x_pred; x_pred itself is
% never evaluated as a direct population member.  After all runs finish,
% the script automatically compares this policy against the completed
% direct-injection K study using paired random seeds.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));

% The completed direct-injection control experiment, if it has also been
% copied to this computer.  Its absence must not prevent the new auxiliary-
% only experiment from running.
directRoot = fullfile(projectRoot,'results_prediction_injection_K', ...
    '20261028_174042');
hasDirectControl = exist(fullfile(directRoot,'injection_runs.csv'),'file') == 2;

% This fixed folder makes an interrupted formal run resumable: simply run
% this same script again and completed SMD/K/run MAT files are skipped.
auxiliaryOnlyRoot = fullfile(projectRoot, ...
    'results_prediction_auxiliary_only_K', ...
    'auxiliary_only_20runs');

runSMDPredictionInjection(auxiliaryOnlyRoot,20,1:12,0:8, ...
    'auxiliary_only');

if hasDirectControl
    comparePredictionInjectionPolicies(directRoot,auxiliaryOnlyRoot);
    fprintf('\nFormal auxiliary-only experiment and paired comparison completed.\n');
    fprintf('Comparison results: %s\n', ...
        fullfile(auxiliaryOnlyRoot,'comparison_with_direct'));
else
    warning(['The direct-injection control folder is not available on this ' ...
        'computer, so only the auxiliary-only experiment was run. Copy ' ...
        'results_prediction_injection_K/20261028_174042 here later, then ' ...
        'run comparePredictionInjectionPolicies(directRoot,auxiliaryOnlyRoot).']);
    fprintf('\nFormal auxiliary-only experiment completed.\n');
end
fprintf('Auxiliary-only results: %s\n',auxiliaryOnlyRoot);
