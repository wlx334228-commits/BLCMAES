function state = predictionModelQualityCapture(action,varargin)
%PREDICTIONMODELQUALITYCAPTURE Read-only snapshots; never fits or evaluates.
switch action
    case 'initialize'
        BI=varargin{1};
        enabled=isfield(BI,'predictionModelQualityDiagnostic') && ...
            isfield(BI.predictionModelQualityDiagnostic,'enabled') && ...
            logical(BI.predictionModelQualityDiagnostic.enabled);
        state=struct('enabled',enabled,'history',zeros(0,BI.u_dim+3), ...
            'events',{{}},'bestUF',-Inf,'historyGenerations',10);
    case 'append'
        [state,CMA,POP,normalPOP,extra,elite,generation,UFE,LFE,updated] = deal(varargin{:});
        upperDim=numel(POP(1).UX);
        rows=zeros(numel(POP),upperDim+3);
        for i=1:numel(POP)
            rows(i,:)=[generation,i,POP(i).UX,POP(i).UF];
        end
        if ~isempty(extra)
            rows(end+1,:)=[generation,numel(POP)+1,extra.UX,extra.UF];
        end
        state.history=[state.history;rows];
        state.history=state.history(state.history(:,1)>= ...
            generation-state.historyGenerations+1,:);
        state.bestUF=max([state.bestUF,[POP.UF],elite.UF]);
        state.events{end+1,1}=struct('generation',generation,'CMA',CMA, ...
            'history',state.history,'normalPopulation',normalPOP, ...
            'populationAfterRefinement',POP,'elite',elite,'UFE',UFE, ...
            'LFE',LFE,'bestUF',state.bestUF,'updateApplied',updated);
    otherwise
        error('BLCMAES:ModelQualityCapture','Unknown capture action.');
end
end
