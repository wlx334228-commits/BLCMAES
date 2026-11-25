function experiment = runSMDSmallAuxiliaryScaleFullComparison( ...
        outputRoot,numberOfRuns,problemNumbers)
%RUNSMALLAUXILIARYSCALEFULLCOMPARISON Terminal-metric small-eta study.
% Compares five complete, paired BL-CMA-ES runs at (DU,DL)=(2,3):
%   A0          8 normal CMA-ES candidates;
%   K3_eta010   5 normal + 1 direct prediction + 2 auxiliary samples;
%   K3_eta015   same K3 composition, eta = 0.15;
%   K3_eta020   same K3 composition, eta = 0.20;
%   K3_eta025   same K3 composition, eta = 0.25.
%
% Every variant runs independently to the unchanged BL-CMA-ES stopping
% rule.  The output files therefore contain final upper/lower errors and
% total upper/lower evaluation counts for every SMD-run pair.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot, ...
        'results_small_auxiliary_scale_full_comparison_2D', ...
        'SMD1_to_SMD8_20runs');
end
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 20; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = 1:8; end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','positive'});
validateattributes(problemNumbers,{'numeric'}, ...
    {'vector','integer','>=',1,'<=',8});
assert(numel(unique(problemNumbers)) == numel(problemNumbers), ...
    'problemNumbers must not contain duplicates.');

studyOptions = struct('etas',[.10 .15 .20 .25],'includeA1',false);
experiment = runSMDFixedAuxiliaryScaleComparison(outputRoot,numberOfRuns, ...
    problemNumbers,studyOptions);

% Use explicit terminal-metric filenames at the top level so the intended
% report cannot be confused with a one-generation diagnostic table.
writetable(experiment.runs,fullfile(outputRoot,'final_scale_runs.csv'));
writetable(experiment.summary,fullfile(outputRoot,'final_scale_summary.csv'));
writetable(experiment.pairedTests, ...
    fullfile(outputRoot,'final_scale_paired_wilcoxon.csv'));
end
