function result = runRobustScaleOutcomeReanalysis(sourceRoot,outputRoot)
%RUNROBUSTSCALEOUTCOMEREANALYSIS Reanalyse completed eta branches only.
% This runner never calls BLCMAES.  It reads the matched branch outcomes
% created by runSMDScaleOutcomeModelExperiment and produces a separate,
% robust analysis folder beside the original experiment.
%
% Current completed experiment:
%   runRobustScaleOutcomeReanalysis
%
% Explicit source/output folders:
%   runRobustScaleOutcomeReanalysis(sourceRoot,outputRoot)

projectRoot = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(sourceRoot)
    sourceRoot = fullfile(projectRoot,'results_scale_outcome_model_2D', ...
        'SMD1_SMD3_SMD6_SMD8_10runs_R3_P2_H5');
end
if nargin < 2 || isempty(outputRoot)
    outputRoot = fullfile(sourceRoot,'robust_reanalysis_v1');
end
assert(exist(fullfile(sourceRoot,'intervention_branch_outcomes.csv'),'file') == 2, ...
    'Missing intervention_branch_outcomes.csv in %s.',sourceRoot);
assert(exist(fullfile(sourceRoot,'state_snapshots.csv'),'file') == 2, ...
    'Missing state_snapshots.csv in %s.',sourceRoot);
if exist(outputRoot,'dir') ~= 7, mkdir(outputRoot); end

outcomes = readtable(fullfile(sourceRoot,'intervention_branch_outcomes.csv'));
snapshots = readtable(fullfile(sourceRoot,'state_snapshots.csv'));
result = analyseRobustScaleOutcomeModels(outcomes,snapshots);

writeResultTables(result,outputRoot);
save(fullfile(outputRoot,'robust_scale_outcome_reanalysis.mat'), ...
    'result','sourceRoot','outputRoot','-v7');
fprintf('Robust scale-outcome reanalysis saved in %s\n',outputRoot);
end

function writeResultTables(result,root)
writetable(result.auditTable,fullfile(root,'robust_reanalysis_audit.csv'));
writetable(result.stateScaleTable,fullfile(root,'robust_state_scale_reference.csv'));
writetable(result.collapsedOutcomes,fullfile(root,'robust_state_eta_outcomes.csv'));
writetable(result.parameterTable,fullfile(root,'robust_model_parameters.csv'));
writetable(result.predictionTable,fullfile(root,'robust_model_predictions.csv'));
writetable(result.intermediateTable,fullfile(root,'robust_intermediate_validation.csv'));
writetable(result.metricTable,fullfile(root,'robust_prediction_metrics.csv'));
writetable(result.metricSummary,fullfile(root,'robust_prediction_metric_summary.csv'));
writetable(result.selectionTable,fullfile(root,'robust_strategy_selection.csv'));
writetable(result.runStrategySummary,fullfile(root,'robust_run_strategy_summary.csv'));
writetable(result.strategySummary,fullfile(root,'robust_strategy_summary.csv'));
writetable(result.pairedDifferences,fullfile(root,'robust_paired_differences.csv'));
writetable(result.pairedTests,fullfile(root,'robust_paired_wilcoxon.csv'));
end
