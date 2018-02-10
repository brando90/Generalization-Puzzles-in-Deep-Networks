%% ranges
lb_x=-100;N_x=100;ub_x=-20;
lb_y=2;N_y=100;ub_y=8;
% lb_x=-5;step_x=0.2;ub_x=5;
% lb_y=-5;step_y=0.2;ub_y=5;
x=linspace(lb_x,ub_x,N_x);
y=linspace(lb_y,ub_y,N_y);
%%
xlab = 'weight w_1';
ylab = 'weight w_2';
% xlab = 'x';
% ylab = 'y';
%% Slope itself
[X,Y] = meshgrid(x,y);
Z_slope = 10*slope(X);
fig = figure;
surf(X,Y,Z_slope)
title('Slope');
xlabel(xlab)
ylabel(ylab)
%% Plot Slopped Valley
[X,Y] = meshgrid(x,y);
Z = Z_slope+-exp(-(0.01* abs(X).^0.95 ).*(Y-5).^2 ) + 1;
Z2 = Z_slope+-exp(-(0.1 ).*(Y-5).^2 ) + 1;
fig=figure;
surf(X,Y,Z2)
fig = figure;
surf(X,Y,Z)
title('Energy Landscape');
xlabel(xlab)
ylabel(ylab)
zlabel('Loss')
%% Save files
%save(fig,'sloped_valley');
saveas(fig,'sloped_valley_shrinking');
saveas(fig,'sloped_valley_shrinking','pdf');
function x = slope(X)
x = -1./(X*0.5-2.8);
end
function w = weighting(Y)
w = exp(-(1/0.1)*Y.^2);
end