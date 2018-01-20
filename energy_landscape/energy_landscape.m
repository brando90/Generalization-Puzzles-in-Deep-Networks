lb_x=-2;N_x=100;ub_x=2;
lb_y=-3;N_y=100;ub_y=3;
% lb_x=-5;step_x=0.2;ub_x=5;
% lb_y=-5;step_y=0.2;ub_y=5;
x=linspace(lb_x,ub_x,N_x);
y=linspace(lb_y,ub_y,N_y);
%%
[X,Y] = meshgrid(x,y);
Z = -exp(-(0.25)*Y.^2) + 1;
%%
fig = figure;
surf(X,Y,Z)
title('Energy Landscape')
xlabel('weight w_1')
ylabel('weight w_2')
zlabel('Loss')
%%
saveas(fig,'energy_landscape');
saveas(fig,'energy_landscape','pdf');