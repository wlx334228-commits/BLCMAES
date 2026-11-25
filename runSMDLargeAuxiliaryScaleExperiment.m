function experiment = runSMDLargeAuxiliaryScaleExperiment( ...
        outputRoot,numberOfRuns,problemNumbers)
%RUNSMDLARGEAUXILIARYSCALEEXPERIMENT Complete enlarged auxiliary-scale study.
% A0: 8 normal candidates. A1: 7 normal + 1 direct prediction.
% K3: 5 normal + 1 direct + 2 auxiliary candidates with
% eta = 1, 1.5, 2, 2.5, 3, 3.5, 4. No auxiliary-only A2 arm.
% All variants run to the original stopping condition at (DU,DL)=(2,3).
% Defaults: SMD1--SMD8, 20 paired seeds; completed leaf runs are skipped.
%
%   runSMDLargeAuxiliaryScaleExperiment
%   runSMDLargeAuxiliaryScaleExperiment([],1,1) % one-run SMD1 batch

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot, ...
        'results_large_auxiliary_scale_full_comparison_2D', ...
        'nine_variants_SMD1_to_SMD8_20runs');
end
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 20; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = 1:8; end
options = struct('etas',[1 1.5 2 2.5 3 3.5 4], ...
    'includeA1',true,'includeA2',false, ...
    'comparisonReferenceEta',1,'significanceMode','raw');
experiment = runSMDFixedAuxiliaryScaleComparison( ...
    outputRoot,numberOfRuns,problemNumbers,options);
writetable(experiment.runs,fullfile(outputRoot,'final_scale_runs.csv'));
writetable(experiment.summary,fullfile(outputRoot,'final_scale_summary.csv'));
writetable(experiment.pairedTests, ...
    fullfile(outputRoot,'final_scale_paired_wilcoxon.csv'));
[experiment.generationDiagnostics,experiment.runDiagnostics] = ...
    collectLargeAuxiliaryScaleDiagnostics(outputRoot,experiment.runs);
save(fullfile(outputRoot,'large_scale_experiment_index.mat'),'experiment','-v7');
end
