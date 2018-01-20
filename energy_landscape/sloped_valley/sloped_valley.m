%% ranges
lb_x=-2;N_x=100;ub_x=2;
lb_y=-3;N_y=100;ub_y=3;
% lb_x=-5;step_x=0.2;ub_x=5;
% lb_y=-5;step_y=0.2;ub_y=5;
x=linspace(lb_x,ub_x,N_x);
y=linspace(lb_y,ub_y,N_y);
%%
xlab = 'weight w_1';
ylab = 'weight w_2';
% xlab = 'x';
% ylab = 'y';
%% Weighting for the sloped
[X,Y] = meshgrid(x,y);
Z = weighting(Y);
fig = figure;
surf(X,Y,Z)
title('weighting');
xlabel(xlab)
ylabel(ylab)
%% Slope itself
[X,Y] = meshgrid(x,y);
Z = slope(X);
fig = figure;
surf(X,Y,Z)
title('Slope');
xlabel(xlab)
ylabel(ylab)
%% weight * slope
[X,Y] = meshgrid(x,y);
Z = slope(X).*weighting(Y);
fig = figure;
surf(X,Y,Z)
title('weight*slope');
xlabel(xlab)
ylabel(ylab)
%% weight * slope + valley
[X,Y] = meshgrid(x,y);
Z = slope(X).*weighting(Y) + -exp(-(0.2)*Y.^2) + 1;
fig = figure;
surf(X,Y,Z)
title('weight*slope + valley');
xlabel(xlab)
ylabel(ylab)
%% Plot Slopped Valley
[X,Y] = meshgrid(x,y);
Z = slope(X)+-exp(-(0.2)*Y.^2) + 1;
fig = figure;
surf(X,Y,Z)
title('Energy Landscape');
xlabel(xlab)
ylabel(ylab)
zlabel('Loss')
%% Save files
%save(fig,'sloped_valley');
saveas(fig,'sloped_valley');
saveas(fig,'sloped_valley','pdf');
function x = slope(X)
x = -1./(X.*0.5-2.8);
end
function w = weighting(Y)
w = exp(-(1/0.1)*Y.^2);
end