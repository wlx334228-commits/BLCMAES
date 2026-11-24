function report = testFixedAuxiliaryScaleComparison(outputRoot)
%TESTFIXEDAUXILIARYSCALECOMPARISON Small end-to-end structural check.
% It runs one paired SMD1 replication of the exact four formal variants.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot, ...
        'results_fixed_auxiliary_scale_comparison_tests','SMD1_1run');
end
experiment = runSMDFixedAuxiliaryScaleComparison(outputRoot,1,1);
runs = experiment.runs;
expected = {'A0','A1','K3_eta025','K3_eta050'};
assert(height(runs) == numel(expected), ...
    'The one-run smoke experiment must return exactly four records.');
assert(isequal(sort(runs.variant)',sort(expected)), ...
    'The smoke experiment has an unexpected variant set.');
assert(numel(unique(runs.seed)) == 1, ...
    'All four variants must use one paired seed.');
assert(all(runs.consistency_passed ~= 0), ...
    'A formal variant failed its consistency check.');

for i = 1:numel(expected)
    row = runs(strcmp(runs.variant,expected{i}),:);
    switch expected{i}
        case 'A0'
            expectedComposition = [8 0 0 0];
        case 'A1'
            expectedComposition = [1 7 1 0];
        case {'K3_eta025','K3_eta050'}
            expectedComposition = [3 5 1 2];
    end
    observed = [row.prediction_inject_count, ...
        row.configured_normal_candidate_count, ...
        row.configured_direct_candidate_count, ...
        row.configured_auxiliary_candidate_count];
    assert(isequal(observed,expectedComposition), ...
        '%s has an invalid population composition.',expected{i});
end

assert(exist(fullfile(outputRoot,'fixed_scale_runs.csv'),'file') == 2 && ...
    exist(fullfile(outputRoot,'fixed_scale_summary.csv'),'file') == 2 && ...
    exist(fullfile(outputRoot,'fixed_scale_paired_wilcoxon.csv'),'file') == 2 && ...
    exist(fullfile(outputRoot,'fixed_scale_matched_budget_summary.csv'),'file') == 2, ...
    'The fixed-scale result tables were not written.');
report = struct('outputRoot',outputRoot,'variants',{expected}, ...
    'seed',runs.seed(1),'runCount',height(runs),'passed',true);
save(fullfile(outputRoot,'fixed_scale_test_report.mat'),'report');
disp(report);
end
