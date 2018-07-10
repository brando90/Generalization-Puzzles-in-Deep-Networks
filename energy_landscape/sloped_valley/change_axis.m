clear
close all
fig = openfig('sloped_valley_shrinking_with_w_norm.fig');
ax = findobj(fig, 'type', 'axes', '-or', 'type', 'matlab.graphics.axis.Axes')
%ax = findobj(0, 'type', 'axes', '-or', 'type', 'matlab.graphics.axis.Axes');
if isempty(ax); error('No axes found'); end
if length(ax) > 1; error('Multiple axes found'); end
xticks(ax, [-100 -80 -60 -40 -20])
set(ax, 'XTickLabel', {'6' '5' '4' '3' '2' '1'});
% yticks(ax, [1])
% set(ax, 'YTickLabel', {'0' '1' '2' '3'});