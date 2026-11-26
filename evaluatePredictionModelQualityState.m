function [details,result] = evaluatePredictionModelQualityState( ...
        event,nextEvent,BI,options,smdNo,runNo,seed,stateIndex,finalUFE,firstUFE)
%EVALUATEPREDICTIONMODELQUALITYSTATE Offline, matched true lower evaluations.
% The caller's counters AND global RNG are restored, including on errors.
global ulFunctionEvaluations llFunctionEvaluations;
savedU=ulFunctionEvaluations; savedL=llFunctionEvaluations; savedRng=rng;
cleanup=onCleanup(@() restoreGlobals(savedU,savedL,savedRng)); %#ok<NASGU>
poolSeed=mod(seed*104729+event.generation*8191+271828,2^31-2)+1;
lowerSeed=mod(seed*130363+event.generation*65537+314159,2^31-2)+1;
normal=nextEvent.normalPopulation;
normalUF=[normal.UF];
[models,shared]=fitPredictionQualityModels(event,BI,options,poolSeed,normal);
evaluationState=struct('cma_mean',event.CMA.xmean,'sigma',event.CMA.sigma, ...
    'C',event.CMA.C,'pc',event.CMA.pc);
rows=repmat(emptyRow(),numel(models),1); evaluations=cell(numel(models),1);
progress=max(0,min(1,(event.UFE-firstUFE)/max(1,finalUFE-firstUFE)));
stage=min(3,1+floor(3*progress));
for i=1:numel(models)
    m=models(i); row=emptyRow();
    row.SMD=smdNo; row.DU=BI.u_dim; row.DL=BI.l_dim;
    row.run_no=runNo; row.seed=seed; row.generation=event.generation;
    row.target_generation=nextEvent.generation; row.state_index=stateIndex;
    row.model=string(m.id); row.stage=stage; row.progress=progress;
    row.main_UFE=event.UFE; row.main_LFE=event.LFE;
    row.cma_mean=encoded(event.CMA.xmean); row.sigma=event.CMA.sigma;
    row.C=encoded(event.CMA.C); row.pc=encoded(event.CMA.pc); row.ps=encoded(event.CMA.ps);
    row.history_sample_count=size(shared.selectedSamples,1);
    row.sample_indices=encoded(shared.selectedSamples(:,1:2));
    row.selected_samples=encoded(shared.selectedSamples);
    row.local_coordinate_scale=shared.scale; row.local_UF_mean=shared.yMean;
    row.local_UF_std=shared.yScale; row.search_radius_normalized=shared.radius;
    row.candidate_pool_seed=poolSeed; row.lower_evaluation_seed=lowerSeed;
    row.common_candidate_count=size(shared.candidateZ,1);
    row.fit_success=m.fitValid; row.prediction_valid=m.valid;
    row.failure_reason=string(m.reason); row.failure_message=string(m.failureMessage);
    row.fit_seconds=m.fitTime; row.search_seconds=m.optimizationTime;
    row.fit_reuses_QP=m.id=="QP_analytic";
    row.fit_RMSE=m.fitRMSE; row.design_condition=m.designCondition;
    row.regularization=m.regularization; row.gpr_mean_cache_error=m.gpMeanCheckError;
    row.gpr_warnings=string(m.gpWarnings);
    row.optimizer_exit_flags=encoded(m.exitFlags);
    row.optimizer_function_counts=encoded(m.optimizerFunctionCounts);
    row.optimizer_successful_starts=m.successfulStarts;
    row.surrogate_evaluations=m.surrogateEvaluations;
    row.duplicate_history=m.duplicateHistory; row.duplicate_earlier_model=m.duplicateEarlierModel;
    row.duplicate_optimizer_solutions=m.duplicateOptimizerSolutions;
    row.boundary_repaired=m.boundaryRepaired; row.at_search_boundary=m.atBoundary;
    row.inside_common_region=m.insideCommonRegion;
    row.current_best_UF=event.bestUF; row.current_best_upper_objective=-event.bestUF;
    row.next_normal_UF=encoded(normalUF); row.next_normal_upper_objectives=encoded(-normalUF);
    row.next_normal_UX=encoded(vertcat(normal.UX)); row.next_normal_LX=encoded(vertcat(normal.LX));
    row.next_normal_LF=encoded([normal.LF]);
    row.next_normal_best_UF=max(normalUF); row.next_normal_median_UF=median(normalUF);
    row.next_normal_best_upper_objective=-max(normalUF);
    row.next_normal_median_upper_objective=-median(normalUF);
    row.normal_count=numel(normal); row.mu=event.CMA.mu;
    if m.valid
        row.x_pred=encoded(m.x); row.x_pred_requested=encoded(m.requestedX);
        row.x_pred1=m.x(1); row.x_pred2=m.x(2);
        row.predicted_UF=m.predictedUF; row.predicted_upper_objective=-m.predictedUF;
        row.GPR_predicted_std=m.predictedStd;
        row.distance_to_mean=norm(m.x-shared.mean);
        row.local_normalized_distance=norm(m.z);
        [row.normalized_search_distance,row.distance_covariance_regularized]= ...
            searchDistance(m.x-shared.mean,event.CMA,BI.u_dim);
        % All SMD1--8 have xu*=0 at (2+3); diagnostic only, after prediction.
        row.distance_to_known_optimum=norm(m.x);
        clock=tic;
        beforeU=ulFunctionEvaluations; beforeL=llFunctionEvaluations;
        try
            stream=RandStream('mt19937ar','Seed',lowerSeed);
            evaluation=evaluatePredictionScaleDiagnosticCandidate(m.x,evaluationState,BI,stream);
            evaluations{i}=evaluation;
            row.evaluation_valid=all(isfinite([evaluation.UF,evaluation.LF,evaluation.xl]));
            row.true_UF=evaluation.UF; row.true_upper_objective=-evaluation.UF;
            row.true_LF=evaluation.LF; row.true_lower_objective=-evaluation.LF;
            row.lower_solution=encoded(evaluation.xl); row.lower_RF=evaluation.RF;
            row.upper_error=abs(evaluation.UF-BI.u_fopt);
            row.lower_error_to_joint_optimum=abs(evaluation.LF-BI.l_fopt);
            conditionalOptimum=conditionalLowerOptimum(m.x,smdNo);
            row.lower_conditional_optimal_UF=conditionalOptimum;
            row.lower_conditional_error=abs(evaluation.LF-conditionalOptimum);
            row.lower_precision_passed=row.lower_conditional_error<=BI.l_ftol;
            if row.evaluation_valid
                row.better_than_normal_best=evaluation.UF>max(normalUF);
                row.better_than_normal_median=evaluation.UF>median(normalUF);
                row.better_than_current_best=evaluation.UF>event.bestUF;
                row.beats_normal_pair_rate=mean(evaluation.UF>normalUF);
                row.rank=1+sum(normalUF>evaluation.UF)+.5*sum(normalUF==evaluation.UF);
                row.normalized_rank=(row.rank-1)/numel(normalUF);
                row.potential_parent=row.rank<=event.CMA.mu;
                row.improvement_over_normal_best=evaluation.UF-max(normalUF);
                row.improvement_over_normal_median=evaluation.UF-median(normalUF);
                row.improvement_over_current_best=evaluation.UF-event.bestUF;
            end
        catch exception
            row.failure_reason=string(['evaluation_failed:' exception.identifier]);
            row.failure_message=string(exception.message);
        end
        row.true_evaluation_seconds=toc(clock);
        row.extra_UFE=ulFunctionEvaluations-beforeU;
        row.extra_LFE=llFunctionEvaluations-beforeL;
    end
    rows(i)=row;
end
qp=find(strcmp({models.id},'QP'),1);
for i=1:numel(models)
    if rows(i).evaluation_valid && rows(qp).evaluation_valid
        rows(i).improvement_vs_QP=rows(i).true_UF-rows(qp).true_UF;
        rows(i).distance_to_QP=norm(models(i).x-models(qp).x);
    end
end
details=struct2table(rows);
result=struct('generation',event.generation,'shared',shared,'models',models, ...
    'evaluations',{evaluations},'nextNormalPopulation',normal,'CMA',event.CMA, ...
    'bestUFStart',event.bestUF,'poolSeed',poolSeed,'lowerSeed',lowerSeed);
end

function row=emptyRow()
row=struct('SMD',NaN,'DU',2,'DL',3,'run_no',NaN,'seed',NaN, ...
    'generation',NaN,'target_generation',NaN,'state_index',NaN, ...
    'model',"",'stage',NaN,'progress',NaN,'main_UFE',NaN,'main_LFE',NaN, ...
    'cma_mean',"",'sigma',NaN,'C',"",'pc',"",'ps',"", ...
    'history_sample_count',0,'sample_indices',"",'selected_samples',"", ...
    'local_coordinate_scale',NaN,'local_UF_mean',NaN,'local_UF_std',NaN, ...
    'search_radius_normalized',NaN,'candidate_pool_seed',NaN,'lower_evaluation_seed',NaN, ...
    'common_candidate_count',0,'fit_success',false,'prediction_valid',false, ...
    'evaluation_valid',false,'failure_reason',"",'failure_message',"", ...
    'x_pred',"",'x_pred_requested',"",'x_pred1',NaN,'x_pred2',NaN, ...
    'predicted_UF',NaN,'predicted_upper_objective',NaN,'true_UF',NaN, ...
    'true_upper_objective',NaN,'true_LF',NaN,'true_lower_objective',NaN, ...
    'lower_solution',"",'lower_RF',false,'upper_error',NaN, ...
    'lower_error_to_joint_optimum',NaN,'lower_conditional_optimal_UF',NaN, ...
    'lower_conditional_error',NaN,'lower_precision_passed',false, ...
    'distance_to_mean',NaN,'local_normalized_distance',NaN, ...
    'normalized_search_distance',NaN,'distance_covariance_regularized',false, ...
    'current_best_UF',NaN,'current_best_upper_objective',NaN, ...
    'next_normal_UF',"",'next_normal_upper_objectives',"", ...
    'next_normal_UX',"",'next_normal_LX',"",'next_normal_LF',"", ...
    'next_normal_best_UF',NaN,'next_normal_median_UF',NaN, ...
    'next_normal_best_upper_objective',NaN,'next_normal_median_upper_objective',NaN, ...
    'normal_count',0,'mu',NaN,'better_than_normal_best',false, ...
    'better_than_normal_median',false,'better_than_current_best',false, ...
    'beats_normal_pair_rate',NaN,'rank',NaN,'normalized_rank',NaN, ...
    'potential_parent',false,'improvement_over_normal_best',NaN, ...
    'improvement_over_normal_median',NaN,'improvement_over_current_best',NaN, ...
    'improvement_vs_QP',NaN,'distance_to_QP',NaN,'distance_to_known_optimum',NaN, ...
    'GPR_predicted_std',NaN,'fit_seconds',0,'fit_reuses_QP',false,'search_seconds',0, ...
    'true_evaluation_seconds',0,'extra_UFE',0,'extra_LFE',0,'fit_RMSE',NaN, ...
    'design_condition',NaN,'regularization',NaN,'gpr_mean_cache_error',NaN, ...
    'gpr_warnings',"",'optimizer_exit_flags',"",'optimizer_function_counts',"", ...
    'optimizer_successful_starts',0,'surrogate_evaluations',0, ...
    'duplicate_history',false,'duplicate_earlier_model',false, ...
    'duplicate_optimizer_solutions',0,'boundary_repaired',false, ...
    'at_search_boundary',false,'inside_common_region',false);
end

function value=encoded(x)
value=string(jsonencode(x));
end

function [distance,regularized]=searchDistance(offset,CMA,du)
C=(CMA.C(1:du,1:du)+CMA.C(1:du,1:du)')/2;
[V,D]=eig(C); eigenvalues=diag(D);
floorValue=max(realmin,1e-12*max(eigenvalues));
regularized=any(eigenvalues<floorValue);
eigenvalues=max(eigenvalues,floorValue);
distance=norm((offset*V)./sqrt(eigenvalues)')/(CMA.sigma*sqrt(du));
end

function optimum=conditionalLowerOptimum(x,smd)
% Analytic minimum of the implemented SMD lower objective at fixed xu.
% No objective call and no effect on fitting, selection, or main search.
if smd==7
    optimum=-x(1)^3;
elseif smd==8
    optimum=-abs(x(1));
else
    optimum=-x(1)^2;
end
end

function restoreGlobals(U,L,state)
global ulFunctionEvaluations llFunctionEvaluations;
ulFunctionEvaluations=U; llFunctionEvaluations=L; rng(state);
end
