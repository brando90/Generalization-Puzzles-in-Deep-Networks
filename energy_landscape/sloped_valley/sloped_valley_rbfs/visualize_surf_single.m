function [ ] = visualize_surf_single( f,N,lb,ub )
figure;
x = linspace(lb,ub,N);
y = x;
[X,Y] = meshgrid(x,y);
Z = get_Z_from_meshgrid_f(X,Y,f);
surf(X,Y,Z);
ylabel('weight W_2')
xlabel('weight W_1')
zlabel('Loss')
end

