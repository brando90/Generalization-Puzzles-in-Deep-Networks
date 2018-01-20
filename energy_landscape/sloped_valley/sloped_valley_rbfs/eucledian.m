function [ kern_x ] = eucledian( x,t,tt )
% x = [1 x D]
% t = [K x D]
% tt = [1 x K]
% kern_x = [1 x K]
xx = sum(x.^2,2); % (1 x 1)
kern_x = xx+tt-2*(x*t');
end

