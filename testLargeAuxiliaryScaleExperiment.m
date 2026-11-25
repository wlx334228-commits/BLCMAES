function report = testLargeAuxiliaryScaleExperiment(outputRoot)
%TESTLARGEAUXILIARYSCALEEXPERIMENT Nine full SMD1 runs and source checks.
projectRoot = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_large_auxiliary_scale_tests', ...
        'nine_variants_SMD1_1run');
end
experiment = runSMDLargeAuxiliaryScaleExperiment(outputRoot,1,1);
expected = {'A0','A1','K3_eta100','K3_eta150','K3_eta200', ...
    'K3_eta250','K3_eta300','K3_eta350','K3_eta400'};
runs = experiment.runs;
assert(height(runs)==9 && isequal(sort(runs.variant)',sort(expected)));
assert(numel(unique(runs.seed))==1 && all(runs.consistency_passed~=0));
assert(all(isfinite(runs.upper_error) & isfinite(runs.lower_error) & ...
    runs.UFE>0 & runs.LFE_total>0));
for i = 1:numel(experiment.specifications)
    spec = experiment.specifications(i);
    row = runs(strcmp(runs.variant,spec.id),:);
    loaded = load(fullfile(outputRoot,spec.id, ...
        strrep(char(row.mat_file),'/',filesep)),'injectionLog');
    log = loaded.injectionLog;
    valid = find([log.effective_injected_count]>0);
    if spec.injectCount==0
        assert(isempty(valid));
    else
        assert(~isempty(valid),'%s did not exercise valid prediction.',spec.id);
    end
    for j = valid
        e = log(j);
        assert(sum(strcmp(e.source_labels,'normal'))==spec.normalCount && ...
            sum(strcmp(e.source_labels,'predicted_direct'))==spec.directCount && ...
            sum(strcmp(e.source_labels,'predicted_auxiliary'))==spec.auxiliaryCount);
        if spec.auxiliaryCount>0
            target = spec.runnerEta^2 * e.sigma_before_sampling^2 * ...
                e.C_before_sampling(1:2,1:2);
            assert(norm(target-e.auxiliary_upper_target_covariance,'fro') < ...
                1e-10*max(1,norm(target,'fro')));
        end
    end
end
comparisonNames = experiment.comparisons(:,1);
assert(size(experiment.comparisons,1)==21);
for eta=[1.5 2 2.5 3 3.5 4]
    assert(ismember(sprintf('K3_eta%03d_vs_K3_eta100',round(100*eta)), ...
        comparisonNames),'A scale lacks its eta=1 comparison.');
end
assert(~ismember('UFE_p_holm',experiment.pairedTests.Properties.VariableNames));
for metric={'upper_error','lower_error','UFE','LFE'}
    m=metric{1}; t=experiment.pairedTests;
    assert(isequal(t.([m '_advantage']),t.([m '_p'])<.05 & ...
        t.(['variant_' m '_median'])<t.(['reference_' m '_median'])));
end
d = experiment.runDiagnostics;
assert(height(d)==9 && all(d.auxiliary_upper_repaired_candidates<=d.auxiliary_count));
assert(all(d.auxiliary_limited_parent_steps<=d.auxiliary_selected_parents));
assert(all(d.auxiliary_count==runs.auxiliary_candidate_total));
for file={'final_scale_runs.csv','final_scale_summary.csv', ...
        'final_scale_paired_wilcoxon.csv','auxiliary_scale_run_diagnostics.csv', ...
        'auxiliary_scale_generation_diagnostics.csv'}
    assert(isfile(fullfile(outputRoot,file{1})), 'Missing %s',file{1});
end
report=struct('outputRoot',outputRoot,'variants',{expected},'passed',true);
save(fullfile(outputRoot,'large_auxiliary_scale_test_report.mat'),'report');
disp(report);
end
