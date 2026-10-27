function options = individualPredictionCorrectionOptions(BI)
%INDIVIDUALPREDICTIONCORRECTIONOPTIONS Options for the A0--A4 ablation.
% A0: no transformation; A1: all points toward x_pred; A2: translation
% only; A3: contraction only; A4: two randomly selected points toward
% x_pred.  The old enableIndividualPredictionCorrection flag remains a
% backward-compatible alias for A1.

options = struct('mode','A0','enabled',false,'logEnabled',false, ...
    'weight',0.1,'startGeneration',10,'historyGenerations',10, ...
    'sampleCount',40,'partialCount',2,'partialSelectionSeed',1729, ...
    'minScaleRelative',1e-12,'maxDesignCondition',1e10, ...
    'minHessianRcond',1e-10,'consistencyCheckEnabled',false, ...
    'consistencyTolerance',1e-10);

legacyEnabled = false;
if isfield(BI,'enableIndividualPredictionCorrection')
    validateattributes(BI.enableIndividualPredictionCorrection, ...
        {'logical','numeric'},{'scalar','binary'},mfilename, ...
        'enableIndividualPredictionCorrection');
    legacyEnabled = logical(BI.enableIndividualPredictionCorrection);
end

modeExplicitlySet = false;
if isfield(BI,'individualPredictionCorrection')
    settings = BI.individualPredictionCorrection;
    names = fieldnames(settings);
    for i = 1:numel(names)
        name = names{i};
        if strcmp(name,'enabled') || ~isfield(options,name)
            error('BLCMAES:IndividualPredictionOption', ...
                'Unknown individual-prediction setting: %s',name);
        end
        options.(name) = settings.(name);
        modeExplicitlySet = modeExplicitlySet || strcmp(name,'mode');
    end
end

if legacyEnabled && ~modeExplicitlySet
    options.mode = 'A1';
end
if isstring(options.mode), options.mode = char(options.mode); end
validateattributes(options.mode,{'char'},{'row'});
options.mode = upper(strtrim(options.mode));
validModes = {'A0','A1','A2','A3','A4'};
if ~any(strcmp(options.mode,validModes))
    error('BLCMAES:IndividualPredictionOption', ...
        'mode must be one of A0, A1, A2, A3, or A4.');
end
if legacyEnabled && strcmp(options.mode,'A0')
    error('BLCMAES:IndividualPredictionOption', ...
        'enableIndividualPredictionCorrection=true conflicts with mode A0.');
end
options.enabled = ~strcmp(options.mode,'A0');
options.usesPrediction = any(strcmp(options.mode,{'A1','A2','A4'}));
options.usesTransformation = options.enabled;

validateattributes(options.logEnabled,{'logical','numeric'},{'scalar','binary'});
options.logEnabled = logical(options.logEnabled) || options.enabled;
validateattributes(options.weight,{'numeric'}, ...
    {'real','finite','scalar','>=',0,'<=',1});
for name = {'startGeneration','historyGenerations','sampleCount','partialCount'}
    validateattributes(options.(name{1}),{'numeric'}, ...
        {'real','finite','scalar','integer','positive'});
end
validateattributes(options.partialSelectionSeed,{'numeric'}, ...
    {'real','finite','scalar','integer','nonnegative'});
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
    error('BLCMAES:IndividualPredictionOption', ...
        'A full 2-D quadratic needs at least six samples.');
end
if options.usesPrediction || options.logEnabled
    supported = arrayfun(@(k)sprintf('smd%d',k),1:8,'UniformOutput',false);
    if BI.u_dim ~= 2 || ~any(strcmpi(BI.fn,supported))
        error('BLCMAES:IndividualPredictionScope', ...
            ['Individual prediction ablations support only SMD1--SMD8 ' ...
             'with two upper variables.']);
    end
end
end
