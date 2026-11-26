function report=testPredictionRefinementExperiment(outputRoot)
%TESTPREDICTIONREFINEMENTEXPERIMENT Original P1 regression, full archive, S2.
root=fileparts(mfilename('fullpath')); addpath(root,fullfile(root,'BLEA_D'));
if nargin<1||isempty(outputRoot)
    outputRoot=fullfile(root,'results_prediction_refinement_tests','SMD1_1run');
end
experiment=runSMDPredictionRefinementExperiment(outputRoot,1,1,struct('reuseMainResults',false));
D=experiment.details; S=experiment.samples; eligible=D(D.eligible_opportunity,:);
assert(all(experiment.audit.isolation_verified&experiment.audit.main_trajectory_equal&experiment.audit.main_rng_equal));
assert(all(eligible.history_sample_count==40));
assert(all(S.H2_age_max(S.P1_valid)<=9),'H2 changed the original historical window.');
assert(any(S.H2_new_sample_count(S.P1_valid)>0),'Test did not exercise selection outside H1.');
main=load(fullfile(outputRoot,'SMD1','run001','main_run.mat'),'mainIns','BI');
events=main.mainIns.predictionModelQualityCapture.events; BI=main.BI;
fullArchive=zeros(0,5); exercised=false;
for j=1:numel(events)-1
    event=events{j}; fullArchive=[fullArchive;event.history(event.history(:,1)==event.generation,:)]; %#ok<AGROW>
    if event.generation<10, continue; end
    event.history=fullArchive;
    state=load(fullfile(outputRoot,'SMD1','run001',sprintf('state_gen%04d.mat',event.generation)),'stateResult','part');
    p1=state.stateResult.models(1);
    % Compare directly against the unchanged original stationary-point fit.
    original=fitLocalUpperQuadratic(event.history,event.CMA.xmean(1:2),event.generation,predictionInjectionOptions(BI));
    assert(original.valid==state.part.eligible_opportunity(1));
    if ~original.valid, continue; end
    requested=original.x_pred; reference=events{j+1}.normalPopulation(end).UX;
    high=requested>BI.u_ub; requested(high)=(reference(high)+BI.u_ub(high))/2;
    low=requested<BI.u_lb; requested(low)=(reference(low)+BI.u_lb(low))/2;
    assert(isequal(requested,p1.x),'Original P1 prediction changed.');
    H2=state.stateResult.H2;
    candidates=event.history(event.history(:,1)>=event.generation-9,:);
    dist=vecnorm(candidates(:,3:4)-p1.x,2,2);
    [~,order]=sortrows([dist,(1:numel(dist))'],[1 2]);
    assert(isequal(H2,candidates(order(1:40),:)),'H2 is not a full-archive nearest-40 selection.');
    for k=find(startsWith(state.part.model,"Q"))'
        m=state.stateResult.models(k);
        if m.fitValid
            assert(m.quadprogExitFlag>0&&all(diag(m.H)<0));
            assert(numel(m.payload.coefficients)==5,'Constrained fit must use five diagonal-model coefficients.');
        end
    end
    if ~exercised
        opts=predictionRefinementOptions(struct('modes',{{'S0','S1','S2'}}));
        before=rng; global ulFunctionEvaluations llFunctionEvaluations;
        U=ulFunctionEvaluations; L=llFunctionEvaluations;
        [local,~,~]=evaluatePredictionRefinementState(event,events{j+1},BI,predictionModelQualityOptions, ...
            opts,1,1,1,1,main.mainIns.UFEs,event.UFE);
        assert(isequal(before,rng)&&isequaln(U,ulFunctionEvaluations)&&isequaln(L,llFunctionEvaluations));
        local=local(local.mode=="S2"&local.prediction_valid,:);
        assert(height(local)>=1,'S2 did not produce any valid local prediction.');
        C=event.CMA.C(1:2,1:2);
        for k=1:height(local)
            offset=[local.x_pred1(k) local.x_pred2(k)]-p1.x;
            assert(sqrt(offset*(C\offset'))/event.CMA.sigma<=opts.localMahalanobisRadius+1e-6);
        end
        % Exact duplicates may not occur naturally in this run. Exercise
        % exact reuse explicitly, and keep a nearby but different point.
        opts=predictionRefinementOptions;
        prepared=preparePredictionRefinementState(event,events{j+1},BI,predictionModelQualityOptions,opts,1);
        for k=2:numel(prepared.models), prepared.models(k).valid=false; end
        prepared.models(2)=prepared.models(1); prepared.models(2).id='P2';
        prepared.models(3)=prepared.models(1); prepared.models(3).id='G1';
        prepared.models(3).x=prepared.models(3).x+[1e-9 0];
        [reused,~]=evaluatePredictionModelQualityState(event,events{j+1},BI,predictionModelQualityOptions, ...
            1,1,1,1,main.mainIns.UFEs,event.UFE,prepared);
        assert(reused.evaluation_reused(2)&&reused.extra_UFE(2)==0&&reused.extra_LFE(2)==0);
        assert(~reused.evaluation_reused(3)&&sum(reused.extra_UFE)==2, ...
            'Nearby, nonidentical predictions must not be deleted or merged.');
        % Original GPR/RBF fits and default optimizers retain their exact
        % model definitions and selected-point behaviour after extension.
        oldFile=fullfile(root,'results_prediction_model_quality_2D','SMD1_to_SMD8_10runs', ...
            'SMD1','run001',sprintf('state_gen%04d.mat',event.generation));
        if isfile(oldFile)
            old=load(oldFile,'stateResult');
            g=find(strcmp({state.stateResult.models.id},'G1')); r=find(strcmp({state.stateResult.models.id},'R1'));
            assert(norm(state.stateResult.models(g).x-old.stateResult.models(4).x)<1e-10);
            assert(norm(state.stateResult.models(r).x-old.stateResult.models(3).x)<1e-10);
        end
        bad=event;
        bad.history(:,5)=sum((bad.history(:,3:4)-event.CMA.xmean(1:2)).^2,2);
        [gated,~,badSamples]=evaluatePredictionRefinementState(bad,events{j+1},BI, ...
            predictionModelQualityOptions,opts,1,1,1,1,main.mainIns.UFEs,event.UFE);
        assert(~any(gated.eligible_opportunity)&&sum(gated.extra_UFE)==0&&badSamples.sample_count_H2==0);
        % No distance or direction threshold is used for discarding points.
        exercised=true;
    end
end
G=experiment.analysis.geometry;
valid=G.prediction_pair_valid;
assert(all(G.euclidean_distance(valid)>=0&G.mahalanobis_sigma_distance(valid)>=0));
assert(all(abs(G.cosine_raw(isfinite(G.cosine_raw)))<=1+1e-12));
assert(all(abs(G.cosine_whitened(isfinite(G.cosine_whitened)))<=1+1e-12));
assert(all(D.extra_UFE(D.evaluation_reused)==0&D.extra_LFE(D.evaluation_reused)==0));
zero=eligible(eligible.generation==eligible.generation(1)&ismember(eligible.model,["P1","P2"]),:);
meanValue=jsondecode(zero.cma_mean(1));
zero.x_pred1(1)=meanValue(1); zero.x_pred2(1)=meanValue(2);
zero.x_pred(1)=string(jsonencode(meanValue(1:2)));
[zeroGeometry,~]=predictionRefinementRelations(zero);
assert(isnan(zeroGeometry.cosine_raw)&&isnan(zeroGeometry.cosine_whitened), ...
    'Zero-length direction similarity must remain unavailable.');
files={'details.csv','run_summary.csv','summary.csv','pairwise.csv','sample_analysis.csv', ...
    'complementarity.csv','geometry.csv','stage_summary.csv','results.mat'};
for i=1:numel(files), assert(isfile(fullfile(outputRoot,['prediction_refinement_' files{i}]))); end
report=struct('passed',true,'outputRoot',outputRoot,'originalP1Unchanged',true, ...
    'fullArchiveH2Verified',true,'mainTrajectoryIsolated',true,'S2BoundsVerified',exercised, ...
    'allOpportunities',height(S),'eligibleP1States',sum(S.P1_valid), ...
    'modelRecords',height(D),'extraUFE',sum(D.extra_UFE),'extraLFE',sum(D.extra_LFE));
save(fullfile(outputRoot,'prediction_refinement_test_report.mat'),'report'); disp(report);
end
