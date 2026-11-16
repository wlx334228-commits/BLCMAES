function files = plotPredictionScaleAllEventActual(eventTable,candidateTable,outputDir)
%PLOTPREDICTIONSCALEALLEVENTACTUAL Plot real F050 scale trajectories.
% Points are individual realised events/candidates for visualization only;
% formal summaries continue to aggregate within run first.

if exist(outputDir,'dir') ~= 7, mkdir(outputDir); end
files = cell(0,1);
stageNames = {'early','middle','late'};
colours = [0.00 0.45 0.74;0.85 0.33 0.10;0.47 0.67 0.19];
for smdNo = unique(eventTable.SMD)'
    events = eventTable(eventTable.SMD == smdNo,:);
    candidates = candidateTable(candidateTable.SMD == smdNo,:);
    figureHandle = figure('Visible','off','Color','w','Position',[100 100 1250 360]);
    subplot(1,3,1); hold on; grid on;
    subplot(1,3,2); hold on; grid on;
    subplot(1,3,3); hold on; grid on;
    for stageIndex = 1:numel(stageNames)
        stage = stageNames{stageIndex};
        eventSubset = events(strcmp(events.stage,stage),:);
        candidateSubset = candidates(strcmp(candidates.stage,stage),:);
        if ~isempty(eventSubset)
            subplot(1,3,1);
            semilogy(eventSubset.progress,eventSubset.sigma,'o', ...
                'Color',colours(stageIndex,:),'MarkerSize',4, ...
                'DisplayName',stage);
            subplot(1,3,2);
            semilogy(eventSubset.progress,eventSubset.auxiliary_rms_scale,'o', ...
                'Color',colours(stageIndex,:),'MarkerSize',4, ...
                'DisplayName',stage);
        end
        if ~isempty(candidateSubset)
            subplot(1,3,3);
            scatter(candidateSubset.auxiliary_rms_scale, ...
                candidateSubset.delta_UF_to_direct,20,colours(stageIndex,:), ...
                'filled','DisplayName',stage);
        end
    end
    for subplotIndex = 1:2
        subplot(1,3,subplotIndex);
        xline(1/3,'k:'); xline(2/3,'k:'); xlim([0 1]);
        xlabel('Post-run relative progress');
        if subplotIndex == 1
            ylabel('\sigma_t'); title(sprintf('SMD%d: CMA-ES step size',smdNo));
        else
            ylabel('\eta_t\sigma_t sqrt(trace(C_{UU})/D_U)');
            title(sprintf('SMD%d: realised auxiliary RMS scale',smdNo));
        end
    end
    subplot(1,3,3); grid on;
    xlabel('Auxiliary RMS scale'); ylabel('\Delta UF to direct prediction');
    title(sprintf('SMD%d: scale versus auxiliary quality',smdNo));
    legend('Location','best');
    fileName = fullfile(outputDir,sprintf('SMD%d_all_event_scale_quality.png',smdNo));
    saveas(figureHandle,fileName); close(figureHandle);
    files{end+1,1} = fileName; %#ok<AGROW>
end
end
