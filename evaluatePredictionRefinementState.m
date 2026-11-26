function [details,result,samples]=evaluatePredictionRefinementState( ...
        event,nextEvent,BI,qualityOptions,options,smd,runNo,seed,index,endUFE,startUFE)
%EVALUATEPREDICTIONREFINEMENTSTATE Reuses isolated true-evaluation machinery.
global ulFunctionEvaluations llFunctionEvaluations;
beforeRng=rng; beforeU=ulFunctionEvaluations; beforeL=llFunctionEvaluations;
cleanup=onCleanup(@() restore(beforeU,beforeL,beforeRng)); %#ok<NASGU>
clock=tic;
prepared=preparePredictionRefinementState(event,nextEvent,BI,qualityOptions,options,seed);
[details,result]=evaluatePredictionModelQualityState(event,nextEvent,BI,qualityOptions, ...
    smd,runNo,seed,index,endUFE,startUFE,prepared);
details.mode=prepared.mode; details.sample_set=prepared.sampleSet;
details.eligible_opportunity=repmat(prepared.P1valid,height(details),1);
details.P1_failure_reason=repmat(string(prepared.P1reason),height(details),1);
details.improvement_vs_P1=NaN(height(details),1);
details.normalized_improvement_vs_P1=NaN(height(details),1);
details.distance_to_P1=NaN(height(details),1);
floorValue=1e-12*max(1,abs(prepared.H1.yMean));
referenceScale=max(prepared.H1.yScale,floorValue);
details.normalized_refinement_scale=repmat(referenceScale,height(details),1);
details.normalization_floor_applied=repmat(prepared.H1.yScale<floorValue,height(details),1);
details.mahalanobis_sigma_distance=sqrt(BI.u_dim)*details.normalized_search_distance;
details.direction_from_mean=repmat("",height(details),1);
details.H_UF=string(arrayfun(@(m)jsonencode(m.H),prepared.models,'UniformOutput',false));
details.quadprog_exit_flag=[prepared.models.quadprogExitFlag]';
details.strict_curvature_floor=[prepared.models.curvatureFloor]';
details.search_region=repmat("legacy mean-centred local ball and bounds",height(details),1);
for i=1:height(details)
    if details.model(i)=="P1"||details.model(i)=="P2"
        details.search_region(i)="original analytic rule, variable bounds, original distance limit";
    elseif details.mode(i)=="S2"
        details.search_region(i)="original allowed region intersect P1 Mahalanobis neighbourhood";
    end
    if details.prediction_valid(i)
        details.distance_to_P1(i)=norm(prepared.models(i).x-prepared.models(1).x);
        details.direction_from_mean(i)=string(jsonencode(prepared.models(i).x-event.CMA.xmean(1:2)));
    end
    if details.evaluation_valid(i)&&details.evaluation_valid(1)
        details.improvement_vs_P1(i)=details.true_UF(i)-details.true_UF(1);
        details.normalized_improvement_vs_P1(i)=details.improvement_vs_P1(i)/referenceScale;
    end
end
samples=predictionRefinementSampleAnalysis(event,prepared,smd,runNo,seed,index);
samples.stage=details.stage(1); samples.progress=details.progress(1);
result.prepared=prepared; result.fullHistoricalArchive=event.history;
result.H1=prepared.H1.selectedSamples;
if prepared.P1valid, result.H2=prepared.H2.selectedSamples; else, result.H2=zeros(0,5); end
result.state_total_seconds=toc(clock);
details.state_total_seconds=repmat(result.state_total_seconds,height(details),1);
end

function restore(U,L,state)
global ulFunctionEvaluations llFunctionEvaluations;
ulFunctionEvaluations=U; llFunctionEvaluations=L; rng(state);
end
