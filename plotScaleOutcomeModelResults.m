function files = plotScaleOutcomeModelResults(outcomes,modelResult,figureRoot)
%PLOTSCALEOUTCOMEMODELRESULTS Compact visual checks for the offline study.
% Figures are intentionally descriptive; inference remains at the independent
% main-run level in the CSV summaries.

files = {};
if exist(figureRoot,'dir') ~= 7, mkdir(figureRoot); end
if isempty(outcomes), return; end

try
    training = outcomes(outcomes.is_training_eta ~= 0,:);
    if ~isempty(training)
        figure('Visible','off'); hold on;
        phases = unique(string(training.phase),'stable');
        for phaseIndex = 1:numel(phases)
            part = training(string(training.phase) == phases(phaseIndex),:);
            eta = unique(part.eta)'; values = NaN(size(eta));
            for etaIndex = 1:numel(eta)
                values(etaIndex) = median(part.relative_overall_gain( ...
                    abs(part.eta-eta(etaIndex)) < 10*eps),'omitnan');
            end
            plot(eta,values,'-o','LineWidth',1.25, ...
                'DisplayName',char(phases(phaseIndex)));
        end
        yline(0,'k:'); xlabel('\eta'); ylabel('relative overall gain y');
        title('Single-generation intervention outcome by phase'); legend('Location','best');
        grid on; file = fullfile(figureRoot,'eta_outcome_by_phase.png');
        exportgraphics(gcf,file,'Resolution',160); close(gcf); files{end+1,1} = file; %#ok<AGROW>
    end

    if ~isempty(modelResult.selectionTable)
        figure('Visible','off');
        summary = modelResult.strategySummary;
        policy = string(summary.policy);
        bar(summary.actual_y_mean); hold on;
        errorbar(1:height(summary),summary.actual_y_mean,summary.actual_y_std, ...
            '.k','LineWidth',1);
        set(gca,'XTick',1:height(summary),'XTickLabel',policy, ...
            'XTickLabelRotation',35);
        ylabel('run-level mean actual y'); title('Held-out strategy outcome'); grid on;
        file = fullfile(figureRoot,'heldout_strategy_outcomes.png');
        exportgraphics(gcf,file,'Resolution',160); close(gcf); files{end+1,1} = file; %#ok<AGROW>
    end

    stateKeys = unique(string(training.state_key),'stable');
    d = NaN(numel(stateKeys),1); q = d; benefit = d;
    for index = 1:numel(stateKeys)
        part = training(string(training.state_key) == stateKeys(index),:);
        row025 = part(abs(part.eta-0.25) < 10*eps,:);
        row050 = part(abs(part.eta-0.50) < 10*eps,:);
        if ~isempty(row025) && ~isempty(row050)
            d(index) = row025.d(1); q(index) = row025.q(1);
            benefit(index) = row025.relative_overall_gain(1)- ...
                row050.relative_overall_gain(1);
        end
    end
    if any(isfinite(benefit))
        figure('Visible','off');
        scatter(d,q,45,benefit,'filled'); colorbar; grid on;
        xlabel('prediction offset d'); ylabel('local curvature q');
        title('\eta=.25 relative gain versus \eta=.50 by state');
        file = fullfile(figureRoot,'dq_interaction_eta025_vs_eta050.png');
        exportgraphics(gcf,file,'Resolution',160); close(gcf); files{end+1,1} = file; %#ok<AGROW>
    end

    if ~isempty(modelResult.predictionTable)
        figure('Visible','off'); hold on;
        firstState = string(modelResult.predictionTable.state_key(1));
        firstFold = modelResult.predictionTable.fold(1);
        rows = modelResult.predictionTable(string(modelResult.predictionTable.state_key) ...
            == firstState & modelResult.predictionTable.fold == firstFold & ...
            strcmp(modelResult.predictionTable.split,'discrete'),:);
        observed = rows(strcmp(rows.model,'M1'),:);
        if ~isempty(observed)
            observed = sortrows(observed,'eta');
            plot(observed.eta,observed.observed_y,'ko-','LineWidth',1.5, ...
                'DisplayName','true');
        end
        for label = ["M0","M1"]
            part = rows(string(rows.model) == label,:);
            if isempty(part), continue; end
            part = sortrows(part,'eta');
            plot(part.eta,part.predicted_y,'--o','LineWidth',1.2, ...
                'DisplayName',char(label));
        end
        yline(0,'k:'); grid on; xlabel('\eta'); ylabel('relative gain y');
        title('Example held-out state: predicted scale-outcome curves');
        legend('Location','best');
        file = fullfile(figureRoot,'example_heldout_scale_curves.png');
        exportgraphics(gcf,file,'Resolution',160); close(gcf); files{end+1,1} = file; %#ok<AGROW>
    end
catch plotException
    warning('BLCMAES:ScaleOutcomePlot', ...
        'Scale-outcome figures were skipped: %s',plotException.message);
end
end
