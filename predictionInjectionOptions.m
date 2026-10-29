function options = predictionInjectionOptions(BI)
%PREDICTIONINJECTIONOPTIONS Options for K=0,...,lambda prediction injection.
% Legacy A0/B1/B2 names remain accepted: A0=K0, B1=K1, and B2=K2.
% K>=1 replaces K normal candidates by one direct quadratic prediction and
% K-1 private-stream auxiliary samples around that prediction.

options = struct('mode','A0','predictionInjectCount',NaN, ...
    'enabled',false,'logEnabled',false, ...
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
validModes = [{'A0','B1','B2'}, ...
    arrayfun(@(k)sprintf('K%d',k),0:8,'UniformOutput',false)];
if ~any(strcmp(options.mode,validModes))
    error('BLCMAES:PredictionInjectionOption', ...
        'mode must be A0, B1, B2, or K0 through K8.');
end

% A numeric K is the canonical setting for the new ablation.  Do not make
% the legacy B1/B2 scripts change behaviour: when K is omitted, their old
% names still select exactly the old candidate layout and RNG policy.
modeK = modeToInjectCount(options.mode);
if isnan(options.predictionInjectCount)
    options.predictionInjectCount = modeK;
else
    validateattributes(options.predictionInjectCount,{'numeric'}, ...
        {'real','finite','scalar','integer','>=',0,'<=',8});
    if ~strcmp(options.mode,'A0') && options.predictionInjectCount ~= modeK
        error('BLCMAES:PredictionInjectionOption', ...
            'mode %s conflicts with predictionInjectCount=%d.', ...
            options.mode,options.predictionInjectCount);
    end
    if isfield(BI,'predictionInjection') && ...
            ~isfield(BI.predictionInjection,'mode') && strcmp(options.mode,'A0')
        % A caller using only the new numeric parameter still receives a
        % self-describing log label rather than a misleading legacy A0 label.
        options.mode = sprintf('K%d',options.predictionInjectCount);
    end
end
options.enabled = options.predictionInjectCount > 0;
options.usesPrediction = options.enabled;
options.usesAuxiliary = options.predictionInjectCount >= 2;
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

function count = modeToInjectCount(mode)
switch mode
    case 'A0'
        count = 0;
    case 'B1'
        count = 1;
    case 'B2'
        count = 2;
    otherwise
        count = sscanf(mode,'K%d');
end
end
