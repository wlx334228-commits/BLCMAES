function options = predictionCorrectionOptions(BI)
%PREDICTIONCORRECTIONOPTIONS Settings for the optional upper-mean correction.
% UF in this repository is a maximization value: UF = -F_original.
% ``weight'' is the prediction-point coefficient in the endpoint blend:
% m_new = (1-weight) * m_cma + weight * x_pred.
options = struct('enabled',false,'logEnabled',false,'weight',0.5, ...
    'startGeneration',10,'correctionInterval',1, ...
    'historyGenerations',10,'sampleCount',40, ...
    'minScaleRelative',1e-12,'maxDesignCondition',1e10, ...
    'minHessianRcond',1e-10,'limitCorrection',false, ...
    'maxCorrectionSigmaFactor',1);
if isfield(BI,'enablePredictionCorrection')
    validateattributes(BI.enablePredictionCorrection,{'logical','numeric'}, ...
        {'scalar','binary'},mfilename,'enablePredictionCorrection');
    options.enabled = logical(BI.enablePredictionCorrection);
end
if isfield(BI,'predictionCorrection')
    settings = BI.predictionCorrection;
    names = fieldnames(settings);
    for i = 1:numel(names)
        name = names{i};
        if strcmp(name,'enabled') || ~isfield(options,name)
            error('BLCMAES:PredictionOption','Unknown prediction setting: %s',name);
        end
        options.(name) = settings.(name);
    end
end
validateattributes(options.logEnabled,{'logical','numeric'},{'scalar','binary'});
options.logEnabled = logical(options.logEnabled) || options.enabled;
validateattributes(options.weight,{'numeric'}, ...
    {'real','finite','scalar','>=',0,'<=',1});
for name = {'startGeneration','correctionInterval', ...
             'historyGenerations','sampleCount'}
    validateattributes(options.(name{1}),{'numeric'}, ...
        {'real','finite','scalar','integer','positive'});
end
validateattributes(options.minScaleRelative,{'numeric'}, ...
    {'real','finite','scalar','positive'});
validateattributes(options.maxDesignCondition,{'numeric'}, ...
    {'real','finite','scalar','>',1});
validateattributes(options.minHessianRcond,{'numeric'}, ...
    {'real','finite','scalar','>',0,'<',1});
validateattributes(options.limitCorrection,{'logical','numeric'},{'scalar','binary'});
validateattributes(options.maxCorrectionSigmaFactor,{'numeric'}, ...
    {'real','finite','scalar','positive'});
if options.sampleCount < 6
    error('BLCMAES:PredictionOption','A full 2-D quadratic needs at least six samples.');
end
if options.enabled || options.logEnabled
    supported = arrayfun(@(k)sprintf('smd%d',k),1:8,'UniformOutput',false);
    if BI.u_dim ~= 2 || ~any(strcmpi(BI.fn,supported))
        error('BLCMAES:PredictionScope', ...
            'Prediction v1 supports only SMD1--SMD8 with two upper variables.');
    end
end
end
