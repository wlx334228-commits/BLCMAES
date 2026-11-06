function options = predictionInjectionOptions(BI)
%PREDICTIONINJECTIONOPTIONS Options for K=0,...,lambda prediction injection.
% Legacy A0/B1/B2 names remain accepted: A0=K0, B1=K1, and B2=K2.
% candidatePolicy='direct_plus_auxiliary' (the original B1/B2/K study)
% replaces K normal candidates by one direct quadratic prediction and K-1
% private-stream auxiliary samples around that prediction.  The new
% candidatePolicy='auxiliary_only' replaces K normal candidates by K
% private-stream auxiliary samples around the prediction, but never inserts
% the prediction itself as an evaluated individual.

options = struct('mode','A0','predictionInjectCount',NaN, ...
    'candidatePolicy','direct_plus_auxiliary', ...
    'auxiliarySamplingDistribution','cma_gaussian', ...
    'auxiliarySamplingScale',1.0, ...
    'hessianCovarianceRelativeEigenvalueFloor',1e-6, ...
    'hessianCovarianceMaxCondition',1e4, ...
    'enabled',false,'logEnabled',false, ...
    'startGeneration',10,'historyGenerations',10,'useAllHistory',false, ...
    'sampleCount',40, ...
    'auxiliaryRandomSeed',314159,'randomSeed',NaN, ...
    'minScaleRelative',1e-12, ...
    'maxDesignCondition',1e10,'minHessianRcond',1e-10, ...
    'maxPredictionNormalizedDistance',Inf, ...
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
options.candidatePolicy = normaliseCandidatePolicy(options.candidatePolicy);
options.auxiliarySamplingDistribution = ...
    normaliseAuxiliarySamplingDistribution(options.auxiliarySamplingDistribution);
validateattributes(options.auxiliarySamplingScale,{'numeric'}, ...
    {'real','finite','scalar','positive'});
validateattributes(options.hessianCovarianceRelativeEigenvalueFloor, ...
    {'numeric'},{'real','finite','scalar','positive','<',1});
validateattributes(options.hessianCovarianceMaxCondition, ...
    {'numeric'},{'real','finite','scalar','>',1});
if any(strcmp(options.mode,{'B1','B2'})) && ...
        ~strcmp(options.candidatePolicy,'direct_plus_auxiliary')
    error('BLCMAES:PredictionInjectionOption', ...
        'Legacy mode %s requires candidatePolicy=''direct_plus_auxiliary''.', ...
        options.mode);
end
options.usesDirectPrediction = options.enabled && ...
    strcmp(options.candidatePolicy,'direct_plus_auxiliary');
options.usesAuxiliary = options.enabled && ...
    (strcmp(options.candidatePolicy,'auxiliary_only') || ...
     options.predictionInjectCount >= 2);
validateattributes(options.logEnabled,{'logical','numeric'},{'scalar','binary'});
options.logEnabled = logical(options.logEnabled) || options.enabled;
validateattributes(options.startGeneration,{'numeric'}, ...
    {'real','finite','scalar','integer','positive'});
validateattributes(options.historyGenerations,{'numeric'}, ...
    {'real','finite','scalar','integer','positive'});
validateattributes(options.useAllHistory,{'logical','numeric'}, ...
    {'scalar','binary'});
options.useAllHistory = logical(options.useAllHistory);
validateattributes(options.sampleCount,{'numeric'}, ...
    {'real','finite','scalar','integer','positive'});

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
validateattributes(options.maxPredictionNormalizedDistance,{'numeric'}, ...
    {'real','scalar','positive'});
validateattributes(options.consistencyCheckEnabled,{'logical','numeric'}, ...
    {'scalar','binary'});
options.consistencyCheckEnabled = logical(options.consistencyCheckEnabled);
validateattributes(options.consistencyTolerance,{'numeric'}, ...
    {'real','finite','scalar','positive'});
quadraticCoefficientCount = (BI.u_dim + 1) * (BI.u_dim + 2) / 2;
if (options.usesPrediction || options.logEnabled) && ...
        options.sampleCount < quadraticCoefficientCount
    error('BLCMAES:PredictionInjectionOption', ...
        ['A full %d-D quadratic needs at least %d samples; ' ...
         'sampleCount is %d.'],BI.u_dim,quadraticCoefficientCount, ...
        options.sampleCount);
end
if options.usesPrediction || options.logEnabled
    % SMD9--SMD12 have constraints, but their original feasibility-aware
    % comparator remains the sole source of ranking and parent selection.
    supported = arrayfun(@(k)sprintf('smd%d',k),1:12,'UniformOutput',false);
    supportedDimensions = (BI.u_dim == 2 && BI.l_dim == 3) || ...
        (BI.u_dim == 5 && BI.l_dim == 5) || ...
        (BI.u_dim == 10 && BI.l_dim == 10);
    if ~supportedDimensions || ~any(strcmpi(BI.fn,supported))
        error('BLCMAES:PredictionInjectionScope', ...
            ['Prediction injection supports SMD1--SMD12 only at (2+3), ' ...
             '(5+5), or (10+10).']);
    end
end
end

function distribution = normaliseAuxiliarySamplingDistribution(distribution)
% Keep the original full-CMA Gaussian as the default.  The other two
% choices alter only the upper marginal around x_pred; lower coordinates
% retain the original conditional CMA-ES relationship in applyPredictionInjection.
if isstring(distribution), distribution = char(distribution); end
validateattributes(distribution,{'char'},{'row'});
distribution = lower(strtrim(distribution));
switch distribution
    case {'cma_gaussian','cma','gaussian'}
        distribution = 'cma_gaussian';
    case {'isotropic_gaussian','isotropic','isotropic_normal'}
        distribution = 'isotropic_gaussian';
    case {'uniform_ellipsoid','ellipsoid','uniform'}
        distribution = 'uniform_ellipsoid';
    case {'hessian_gaussian','hessian','curvature_gaussian'}
        distribution = 'hessian_gaussian';
    otherwise
        error('BLCMAES:PredictionInjectionOption', ...
            ['auxiliarySamplingDistribution must be ''cma_gaussian'', ' ...
             '''isotropic_gaussian'', ''uniform_ellipsoid'', or ' ...
             '''hessian_gaussian''.']);
end
end

function policy = normaliseCandidatePolicy(policy)
if isstring(policy), policy = char(policy); end
validateattributes(policy,{'char'},{'row'});
policy = lower(strtrim(policy));
switch policy
    case {'direct_plus_auxiliary','direct_plus_aux','direct'}
        policy = 'direct_plus_auxiliary';
    case {'auxiliary_only','aux_only'}
        policy = 'auxiliary_only';
    otherwise
        error('BLCMAES:PredictionInjectionOption', ...
            ['candidatePolicy must be ''direct_plus_auxiliary'' or ' ...
             '''auxiliary_only''.']);
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
