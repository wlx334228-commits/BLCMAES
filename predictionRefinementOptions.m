function options=predictionRefinementOptions(settings)
%PREDICTIONREFINEMENTOPTIONS Independent S0/S1/S2 offline experiment controls.
if nargin<1, settings=struct(); end
options=struct('version',1,'modes',{{'S0','S1'}},'localMahalanobisRadius',1, ...
    'curvatureFloorRelative',1e-6,'reuseExactEvaluations',true, ...
    'reuseMainResults',true,'verifyMainTrajectory',true);
for name=fieldnames(settings)'
    assert(isfield(options,name{1}),'Unknown refinement setting: %s',name{1});
    options.(name{1})=settings.(name{1});
end
options.modes=cellstr(upper(string(options.modes))); options.modes=options.modes(:)';
assert(~isempty(options.modes)&&all(ismember(options.modes,{'S0','S1','S2'})));
validateattributes(options.localMahalanobisRadius,{'numeric'},{'scalar','positive','finite'});
validateattributes(options.curvatureFloorRelative,{'numeric'},{'scalar','positive','finite'});
options.models={'P1'};
if ismember('S1',options.modes), options.models{end+1}='P2'; end
if ismember('S0',options.modes), options.models=[options.models,{'G1','Q1','R1'}]; end
if ismember('S1',options.modes), options.models=[options.models,{'G2','Q2','R2'}]; end
if ismember('S2',options.modes), options.models=[options.models,{'P2_S2','G2_S2','Q2_S2','R2_S2'}]; end
% Stable requested order is also the order used for exact-evaluation reuse.
order={'P1','P2','G1','G2','Q1','Q2','R1','R2','P2_S2','G2_S2','Q2_S2','R2_S2'};
options.models=order(ismember(order,options.models));
end
