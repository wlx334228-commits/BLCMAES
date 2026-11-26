function options = predictionModelQualityOptions(settings)
%PREDICTIONMODELQUALITYOPTIONS Shared, fixed four-model diagnostic settings.
if nargin<1, settings=struct(); end
options=struct('version',1,'sampleCount',40,'historyGenerations',10, ...
    'startGeneration',10,'candidateCount',200,'startCount',5, ...
    'localRadius',1,'ridgeRelative',1e-6,'maxIterations',60, ...
    'maxFunctionEvaluations',250,'gpMaxIterations',60, ...
    'duplicateTolerance',1e-9,'verifyMainTrajectory',true);
names=fieldnames(settings);
for i=1:numel(names)
    assert(isfield(options,names{i}),'Unknown model-quality setting: %s',names{i});
    options.(names{i})=settings.(names{i});
end
assert(options.sampleCount==40 && options.historyGenerations==10 && ...
    options.startGeneration==10,'The historical sample policy must remain unchanged.');
assert(options.candidateCount==200 && options.startCount==5, ...
    'All models require the same 200 candidate / 5 start protocol.');
validateattributes(options.localRadius,{'numeric'},{'scalar','positive','<=',1});
validateattributes(options.ridgeRelative,{'numeric'},{'scalar','positive','finite'});
validateattributes(options.maxIterations,{'numeric'},{'scalar','integer','positive'});
validateattributes(options.maxFunctionEvaluations,{'numeric'},{'scalar','integer','positive'});
end
