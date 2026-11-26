function prepared=preparePredictionRefinementState(event,nextEvent,BI,qualityOptions,options,seed)
%PREPAREPREDICTIONREFINEMENTSTATE Original P1 gate, then full-archive H2 search.
% event.history is the reconstructed COMPLETE historical archive through t.
% The old fitter applies exactly the original ten-generation eligibility filter.
poolSeed=mod(seed*104729+event.generation*8191+271828,2^31-2)+1;
normal=nextEvent.normalPopulation;
[p1,H1]=fitPredictionQualityModels(event,BI,qualityOptions,poolSeed,normal, ...
    struct('modelIds',{{'QP_analytic'}}));
p1.id='P1'; p1.fitTime=H1.selectionAndQPFitSeconds;
models=repmat(p1,numel(options.models),1); modelShared=repmat({H1},numel(models),1);
sampleSet=repmat("H1",numel(models),1); mode=repmat("S0",numel(models),1);
for i=1:numel(models)
    models(i).id=options.models{i};
    if i>1
        models(i).fitValid=false; models(i).valid=false;
        models(i).reason=['P1_invalid:' p1.reason]; models(i).fitTime=0;
        models(i).payload=struct();
        models(i).x=[NaN NaN]; models(i).z=[NaN NaN]; models(i).requestedX=[NaN NaN];
    end
    if contains(options.models{i},'2'), sampleSet(i)="H2"; mode(i)="S1"; end
    if endsWith(options.models{i},'_S2'), mode(i)="S2"; end
    if sampleSet(i)=="H2"
        modelShared{i}.selectedSamples=zeros(0,5);
        modelShared{i}.y=zeros(0,1); modelShared{i}.yMean=NaN; modelShared{i}.yScale=NaN;
    end
end
prepared=struct('models',models,'modelShared',{modelShared},'sampleSet',sampleSet, ...
    'mode',mode,'reuseExactEvaluations',options.reuseExactEvaluations, ...
    'P1valid',p1.valid,'P1reason',p1.reason,'H1',H1,'H2',struct(), ...
    'poolSeed',poolSeed,'nativeP1',p1,'nativeP2',struct());
if ~p1.valid, return; end

% Re-select from the full archive, never from H1. The original fitter keeps
% the same window, sample count, Euclidean metric, and stable tie breaking.
[p2,H2]=fitPredictionQualityModels(event,BI,qualityOptions,poolSeed,normal, ...
    struct('modelIds',{{'QP_analytic'}},'selectionCentre',p1.x,'normalizationMean',p1.x));
p2.fitTime=H2.selectionAndQPFitSeconds; prepared.H2=H2; prepared.nativeP2=p2;
commonMean=event.CMA.xmean(1:2); commonScale=H1.scale;
context=struct('modelIds',{{'GPR','CURV','RBF'}}, ...
    'normalizationMean',commonMean,'normalizationScale',commonScale, ...
    'candidateZ',H1.candidateZ);
fitOptions=qualityOptions; fitOptions.curvatureFloorRelative=options.curvatureFloorRelative;
if ismember('S0',options.modes)
    [base,S0]=safeFit(event,BI,fitOptions,poolSeed,normal,context,p1,H1);
    for i=1:3
        put({'G1','Q1','R1'},i,base(i),S0);
    end
end
if ismember('S1',options.modes)
    p2.id='P2'; p2.z=(p2.x-commonMean)/commonScale;
    p2.insideCommonRegion=norm(p2.z)<=qualityOptions.localRadius+1e-8;
    logH2=H2; logH2.mean=commonMean; logH2.scale=commonScale;
    put({'P2'},1,p2,logH2);
    context.selectionCentre=p1.x;
    [reselected,S1]=safeFit(event,BI,fitOptions,poolSeed,normal,context,p1,logH2);
    for i=1:3, put({'G2','Q2','R2'},i,reselected(i),S1); end
end
if ismember('S2',options.modes)
    C=(event.CMA.C(1:2,1:2)+event.CMA.C(1:2,1:2)')/2;
    [V,D]=eig(C); eigenvalues=diag(D);
    assert(all(isfinite(eigenvalues))&&all(eigenvalues>0),'Invalid upper covariance for S2.');
    metric=commonScale^2/event.CMA.sigma^2 * (V*diag(1./eigenvalues)*V');
    localContext=context; localContext.selectionCentre=p1.x;
    localContext=rmfield(localContext,'candidateZ');
    localContext.localCentreZ=(p1.x-commonMean)/commonScale;
    localContext.localMetricZ=(metric+metric')/2;
    localContext.localRadius=options.localMahalanobisRadius;
    [local,S2]=safeFit(event,BI,fitOptions,poolSeed,normal,localContext,p1,H2);
    for i=1:3, put({'G2_S2','Q2_S2','R2_S2'},i,local(i),S2); end
    % Original P2 has the original variable bounds and mean-distance limit;
    % S2 adds the P1 ellipsoid, and maximizes the SAME fitted quadratic.
    localContext.modelIds={'QP'};
    localContext.selectionCentre=commonMean;
    quadraticEvent=event; quadraticEvent.history=H2.selectedSamples;
    quadraticOptions=fitOptions;
    legacyOptions=predictionInjectionOptions(BI);
    quadraticOptions.localRadius=legacyOptions.maxPredictionNormalizedDistance;
    if ~p2.valid
        constrained=p2; constrained.valid=false; constrained.reason=['P2_fit_invalid:' p2.reason];
        quadraticShared=H2;
    else
        [constrained,quadraticShared]=safeFit(quadraticEvent,BI,quadraticOptions, ...
            poolSeed,normal,localContext,p1,H2);
    end
    put({'P2_S2'},1,constrained(1),quadraticShared);
end
prepared.models=models; prepared.modelShared=modelShared;

    function put(names,index,model,shared)
        position=find(strcmp(options.models,names{index}));
        if isempty(position), return; end
        model.id=names{index};
        % Log distances in the main mean/scale coordinates for every group.
        shared.mean=commonMean; shared.scale=commonScale;
        if model.valid, model.z=(model.x-commonMean)/commonScale; end
        models(position)=model; modelShared{position}=shared;
    end
end

function [models,shared]=safeFit(event,BI,options,seed,normal,context,template,shared)
try
    [models,shared]=fitPredictionQualityModels(event,BI,options,seed,normal,context);
catch exception
    models=repmat(template,numel(context.modelIds),1);
    for i=1:numel(models)
        models(i).id=context.modelIds{i}; models(i).valid=false; models(i).fitValid=false;
        models(i).reason=['shared_region_or_fit_failed:' exception.identifier];
        models(i).failureMessage=exception.message; models(i).fitTime=0;
        models(i).optimizationTime=0; models(i).payload=struct();
        models(i).x=[NaN NaN]; models(i).z=[NaN NaN]; models(i).requestedX=[NaN NaN];
    end
end
end
