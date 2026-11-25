function experiment = runSMDSmallAuxiliaryScaleExperiment( ...
        outputRoot,numberOfRuns,problemNumbers)
%RUNSMDSMALLAUXILIARYSCALEEXPERIMENT Complete small-eta comparison.
% Runs eight independent BL-CMA-ES variants to the original termination
% condition: A0, A1 (direct only), A2 (two auxiliary only, eta=.20),
% and K3 with eta = .10/.15/.25/.50/1.00. Unlike the separate
% next-generation diagnostic, every branch produces terminal upper/lower
% errors, total UFE, total LFE, success rates, and paired statistics.
%
% Default formal setup: SMD1--SMD8, 20 paired runs per variant.
%
% Examples:
%   runSMDSmallAuxiliaryScaleExperiment
%   runSMDSmallAuxiliaryScaleExperiment([],1,1)  % small full smoke batch

if nargin < 1, outputRoot = []; end
if nargin < 2, numberOfRuns = []; end
if nargin < 3, problemNumbers = []; end
experiment = runSMDSmallAuxiliaryScaleFullComparison( ...
    outputRoot,numberOfRuns,problemNumbers);
end
