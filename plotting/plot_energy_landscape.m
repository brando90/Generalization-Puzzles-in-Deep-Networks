[X,Y] = meshgrid(-10:0.1:10,-10:0.1:10);
Z = -exp(-(0.25)*Y.^2)+1;
fig = figure;
surf(X,Y,Z)
zlabel('Loss')
xlabel('weight W_1')
ylabel('weight W_2')
saveas(fig,'energy_landscape','pdf');