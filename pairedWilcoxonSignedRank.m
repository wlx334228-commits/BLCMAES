function test = pairedWilcoxonSignedRank(x,y)
%PAIREDWILCOXONSIGNEDRANK Two-sided paired Wilcoxon test without a toolbox.
% x and y must be paired observations.  MATLAB's exact implementation is
% used when available; otherwise a tie-corrected normal approximation is
% reported so batch experiments remain portable.

x = x(:);
y = y(:);
valid = isfinite(x) & isfinite(y);
difference = x(valid)-y(valid);
difference = difference(difference ~= 0);
test = struct('n',numel(difference),'p',NaN,'h',false, ...
    'Wplus',NaN,'method','not_enough_pairs');
if isempty(difference)
    return;
end

if exist('signrank','file') == 2
    try
        [p,h,stats] = signrank(x(valid),y(valid),'method','approximate');
        test.p = p;
        test.h = h;
        test.Wplus = stats.signedrank;
        test.method = 'signrank_approximate';
        return;
    catch
        % Use the deterministic fallback below if a local installation has
        % an older signrank signature or lacks the statistics package.
    end
end

absoluteDifference = abs(difference);
ranks = tiedRanks(absoluteDifference);
Wplus = sum(ranks(difference > 0));
n = numel(difference);
meanW = n*(n+1)/4;
[~,~,tieGroups] = unique(absoluteDifference);
tieCounts = accumarray(tieGroups,1);
varianceW = n*(n+1)*(2*n+1)/24 - sum(tieCounts.^3-tieCounts)/48;
if varianceW <= 0
    return;
end
continuity = 0.5 * sign(Wplus-meanW);
z = (Wplus-meanW-continuity) / sqrt(varianceW);
test.p = min(1,erfc(abs(z)/sqrt(2)));
test.h = test.p < 0.05;
test.Wplus = Wplus;
test.method = 'tie_corrected_normal_approximation';
end

function ranks = tiedRanks(values)
[sortedValues,order] = sort(values);
ranks = zeros(size(values));
startIndex = 1;
while startIndex <= numel(values)
    endIndex = startIndex;
    while endIndex < numel(values) && ...
            sortedValues(endIndex+1) == sortedValues(startIndex)
        endIndex = endIndex + 1;
    end
    ranks(order(startIndex:endIndex)) = (startIndex+endIndex)/2;
    startIndex = endIndex + 1;
end
end
