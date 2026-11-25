function report = testSmallAuxiliaryScaleFullComparison(outputRoot)
%TESTSMALLAUXILIARYSCALEFULLCOMPARISON One-run end-to-end full-study check.
% Verifies that the eight variants are complete optimizer runs and that the
% terminal metric files, rather than only snapshot diagnostics, are saved.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot, ...
        'results_small_auxiliary_scale_full_comparison_tests','eight_variants_SMD1_1run');
end
experiment = runSMDSmallAuxiliaryScaleFullComparison(outputRoot,1,1);
runs = experiment.runs;
expected = {'A0','A1','A2','K3_eta010','K3_eta015','K3_eta025', ...
    'K3_eta050','K3_eta100'};
assert(height(runs) == numel(expected), ...
    'The one-run full study must return eight terminal run records.');
assert(isequal(sort(runs.variant)',sort(expected)), ...
    'The full study has an unexpected variant set.');
assert(numel(unique(runs.seed)) == 1 && all(runs.consistency_passed ~= 0), ...
    'Full-study variants must have one paired seed and pass consistency checks.');
assert(all(isfinite(runs.upper_error) & isfinite(runs.lower_error) & ...
    runs.UFE > 0 & runs.LFE_total > 0), ...
    'A full-study terminal run lacks final error or evaluation metrics.');
for i = 1:numel(experiment.specifications)
    spec = experiment.specifications(i);
    row = runs(strcmp(runs.variant,spec.id),:);
    loaded = load(fullfile(outputRoot,spec.id, ...
        strrep(char(row.mat_file),'/',filesep)),'injectionLog');
    log = loaded.injectionLog;
    valid = find([log.effective_injected_count] > 0);
    if spec.injectCount == 0
        assert(isempty(valid),'A0 must never inject a prediction candidate.');
    else
        assert(~isempty(valid),'%s never exercised valid prediction.',spec.id);
        for j = valid
            labels = log(j).source_labels;
            assert(sum(strcmp(labels,'normal')) == spec.normalCount && ...
                sum(strcmp(labels,'predicted_direct')) == spec.directCount && ...
                sum(strcmp(labels,'predicted_auxiliary')) == spec.auxiliaryCount, ...
                '%s actual source labels do not match its configuration.',spec.id);
        end
    end
end
assert(strcmp(experiment.specifications(3).candidatePolicy,'auxiliary_only') && ...
    experiment.specifications(3).runnerEta == .20, ...
    'A2 must use auxiliary-only injection at eta=0.20.');
assert(exist(fullfile(outputRoot,'final_scale_runs.csv'),'file') == 2 && ...
    exist(fullfile(outputRoot,'final_scale_summary.csv'),'file') == 2 && ...
    exist(fullfile(outputRoot,'final_scale_paired_wilcoxon.csv'),'file') == 2, ...
    'The terminal-metric result tables were not written.');

report = struct('outputRoot',outputRoot,'variants',{expected}, ...
    'seed',runs.seed(1),'runCount',height(runs),'passed',true);
save(fullfile(outputRoot,'small_auxiliary_scale_full_test_report.mat'),'report');
disp(report);
end
