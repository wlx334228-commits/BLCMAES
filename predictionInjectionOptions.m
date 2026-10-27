function options = predictionInjectionOptions(BI)
%PREDICTIONINJECTIONOPTIONS Options for A0/B1/B2 external-solution injection.
% A0 keeps the original population. B1 replaces one sampled upper point by
% x_pred. B2 replaces two points by x_pred and an auxiliary Gaussian sample.

options = struct('mode','A0','enabled',false,'logEnabled',false, ...
    'startGeneration',10,'historyGenerations',10,'sampleCount',40, ...
    'auxiliaryRandomSeed',314159,'randomSeed',NaN, ...
    'minScaleRelative',1e-12, ...
    'maxDesignCondition',1e10,'minHessianRcond',1e-10, ...
    'consistencyCheckEnabled',false,'consistencyTolerance',1e-10);
if isfield(BI,'predictionInjection')
    settings = BI.predictionInjection;
    names = fieldnames(settings);
    for i = 1:numel(names)
        name = names{i};
        if strcmp(name,'enabled') || ~isfield(options,name)
            error('BLCMAES:PredictionInjectionOption', ...
                'Unknown prediction-injection setting: %s',name);
        end
        options.(name) = settings.(name);
    end
end
if isstring(options.mode), options.mode = char(options.mode); end
validateattributes(options.mode,{'char'},{'row'});
options.mode = upper(strtrim(options.mode));
validModes = {'A0','B1','B2'};
if ~any(strcmp(options.mode,validModes))
    error('BLCMAES:PredictionInjectionOption', ...
        'mode must be A0, B1, or B2.');
end
options.enabled = ~strcmp(options.mode,'A0');
options.usesPrediction = options.enabled;
validateattributes(options.logEnabled,{'logical','numeric'},{'scalar','binary'});
options.logEnabled = logical(options.logEnabled) || options.enabled;
for name = {'startGeneration','historyGenerations','sampleCount'}
    validateattributes(options.(name{1}),{'numeric'}, ...
        {'real','finite','scalar','integer','positive'});
end
validateattributes(options.auxiliaryRandomSeed,{'numeric'}, ...
    {'real','finite','scalar','integer','nonnegative'});
validateattributes(options.randomSeed,{'numeric'}, {'real','scalar'});
if ~isnan(options.randomSeed)
    validateattributes(options.randomSeed,{'numeric'}, ...
        {'finite','integer','nonnegative'});
end
validateattributes(options.minScaleRelative,{'numeric'}, ...
    {'real','finite','scalar','positive'});
validateattributes(options.maxDesignCondition,{'numeric'}, ...
    {'real','finite','scalar','>',1});
validateattributes(options.minHessianRcond,{'numeric'}, ...
    {'real','finite','scalar','>',0,'<',1});
validateattributes(options.consistencyCheckEnabled,{'logical','numeric'}, ...
    {'scalar','binary'});
options.consistencyCheckEnabled = logical(options.consistencyCheckEnabled);
validateattributes(options.consistencyTolerance,{'numeric'}, ...
    {'real','finite','scalar','positive'});
if options.sampleCount < 6
    error('BLCMAES:PredictionInjectionOption', ...
        'A full two-dimensional quadratic needs at least six samples.');
end
if options.usesPrediction || options.logEnabled
    % SMD9--SMD12 have constraints, but their original feasibility-aware
    % comparator remains the sole source of ranking and parent selection.
    supported = arrayfun(@(k)sprintf('smd%d',k),1:12,'UniformOutput',false);
    if BI.u_dim ~= 2 || ~any(strcmpi(BI.fn,supported))
        error('BLCMAES:PredictionInjectionScope', ...
            ['Prediction injection currently supports only SMD1--SMD12 ' ...
             'with two upper variables.']);
    end
end
end
