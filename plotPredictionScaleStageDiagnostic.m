function files = plotPredictionScaleStageDiagnostic(stageSummary,outputDir)
%PLOTPREDICTIONSCALESTAGEDIAGNOSTIC Create stage-by-eta quality curves.
% The curves display independent-run medians with IQR error bars.  Missing
% stages remain missing instead of being relabelled as a later phase.

if exist(outputDir,'dir') ~= 7, mkdir(outputDir); end
files = cell(0,1);
stageNames = {'early','middle','late'};
metrics = { ...
    'Q_rank_median','Q_rank_IQR','Q_{rank}: better than normal CMA-ES'; ...
    'Q_pred_median','Q_pred_IQR','Q_{pred}: better than direct prediction'; ...
    'delta_UF_median','delta_UF_IQR','\Delta UF relative to direct prediction'};
colours = [0.00 0.45 0.74;0.85 0.33 0.10;0.47 0.67 0.19];
for smdNo = unique(stageSummary.SMD)'
    figureHandle = figure('Visible','off','Color','w','Position',[100 100 1150 330]);
    for metricIndex = 1:size(metrics,1)
        subplot(1,3,metricIndex); hold on; grid on;
        for stageIndex = 1:numel(stageNames)
            subset = stageSummary(stageSummary.SMD == smdNo & ...
                strcmp(stageSummary.stage,stageNames{stageIndex}),:);
            if isempty(subset), continue; end
            subset = sortrows(subset,'eta');
            errorbar(subset.eta,subset.(metrics{metricIndex,1}), ...
                subset.(metrics{metricIndex,2})/2,'-o', ...
                'LineWidth',1.4,'MarkerSize',5, ...
                'Color',colours(stageIndex,:), ...
                'DisplayName',stageNames{stageIndex});
        end
        xlabel('\eta'); ylabel(metrics{metricIndex,3});
        title(sprintf('SMD%d',smdNo)); xlim([0.2 1.05]);
        xticks([0.25 0.5 0.75 1.0]);
        if metricIndex == 1, legend('Location','best'); end
    end
    fileName = fullfile(outputDir,sprintf('SMD%d_stage_scale_quality.png',smdNo));
    saveas(figureHandle,fileName); close(figureHandle);
    files{end+1,1} = fileName; %#ok<AGROW>
end
end
