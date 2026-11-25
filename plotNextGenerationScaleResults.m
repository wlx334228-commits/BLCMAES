function files = plotNextGenerationScaleResults(outcomes,modelResult,figureRoot,analysisOptions)
%PLOTNEXTGENERATIONSCALERESULTS Descriptive figures for the all-state study.
% Inference is performed at independent-main-run level by the analysis code;
% these plots do not treat repeated states as independent observations.

if nargin < 4 || isempty(analysisOptions)
    analysisOptions = struct('etas',[.25 .50 .75 1.00], ...
        'referenceVariant','eta_050');
end
etaValues = sort(analysisOptions.etas(:)');
referenceVariant = char(analysisOptions.referenceVariant);

files = {};
if exist(figureRoot,'dir') ~= 7, mkdir(figureRoot); end
rows = outcomes(outcomes.is_scale_branch ~= 0 & outcomes.probe_completed ~= 0,:);
if isempty(rows), return; end
for smd = unique(rows.SMD)'
    part = rows(rows.SMD == smd,:);
    try
        f = figure('Visible','off','Color','w','Position',[100 100 1180 440]);
        subplot(1,2,1);
        values = NaN(size(etaValues)); spread = values;
        for i = 1:numel(etaValues)
            x = part.R_stable(abs(part.eta-etaValues(i)) < 1e-12);
            values(i) = median(x,'omitnan'); spread(i)=iqrFinite(x);
        end
        errorbar(etaValues,values,spread/2,'-o','LineWidth',1.4);
        xlabel('\eta'); ylabel(sprintf('R: matched t+1 top-2 UF gain vs %s',referenceVariant));
        title(sprintf('SMD%d: scale and next-generation normal quality',smd)); grid on;
        subplot(1,2,2);
        scatter(part.d,part.q,20,part.eta,'filled'); colorbar;
        xlabel('standardized prediction offset d'); ylabel('normalized curvature q');
        title('All valid pre-injection states'); grid on;
        file = fullfile(figureRoot,sprintf('SMD%d_nextgen_scale_state.png',smd));
        exportgraphics(f,file,'Resolution',180); close(f); files{end+1,1}=file; %#ok<AGROW>
        f = figure('Visible','off','Color','w','Position',[100 100 1180 430]);
        subplot(1,2,1);
        scatter(part.d,part.R_stable,20,part.eta,'filled'); colorbar; grid on;
        xlabel('d'); ylabel('matched next-generation R');
        title('Offset—scale outcome observations');
        subplot(1,2,2);
        scatter(part.q,part.R_stable,20,part.eta,'filled'); colorbar; grid on;
        xlabel('q'); ylabel('matched next-generation R');
        title('Curvature—scale outcome observations');
        file = fullfile(figureRoot,sprintf('SMD%d_nextgen_dq_response.png',smd));
        exportgraphics(f,file,'Resolution',180); close(f); files{end+1,1}=file; %#ok<AGROW>
    catch plotError
        warning('plotNextGenerationScaleResults:PlotFailed','SMD%d plot skipped: %s',smd,plotError.message);
    end
end
if ~isempty(modelResult.strategySummary)
    try
        f = figure('Visible','off','Color','w','Position',[100 100 980 420]);
        policies = unique(modelResult.strategySummary.policy,'stable');
        smds = unique(modelResult.strategySummary.SMD)';
        values = NaN(numel(smds),numel(policies));
        for i = 1:numel(smds)
            for j = 1:numel(policies)
                item = modelResult.strategySummary(modelResult.strategySummary.SMD == smds(i) & ...
                    strcmp(modelResult.strategySummary.policy,policies{j}),:);
                if ~isempty(item), values(i,j)=item.R_median; end
            end
        end
        bar(values); grid on; xlabel('SMD problem'); ylabel('median selected R');
        set(gca,'XTick',1:numel(smds),'XTickLabel',arrayfun(@(x)sprintf('SMD%d',x),smds,'UniformOutput',false));
        legend(policies,'Interpreter','none','Location','bestoutside');
        title('Cross-validated scale-selector outcomes');
        file=fullfile(figureRoot,'nextgen_strategy_comparison.png');
        exportgraphics(f,file,'Resolution',180); close(f); files{end+1,1}=file; %#ok<AGROW>
    catch plotError
        warning('plotNextGenerationScaleResults:SummaryPlotFailed','Summary plot skipped: %s',plotError.message);
    end
end
end

function value = iqrFinite(x)
x=x(isfinite(x)); if isempty(x), value=NaN; else, value=iqr(x); end
end
