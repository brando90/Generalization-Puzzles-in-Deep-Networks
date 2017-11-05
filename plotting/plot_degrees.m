n=2;
D=20;
degrees = 1:D;
y = zeros(1,D);
for d=degrees
    y(d) = nchoosek(d+n,d);
end
y
figure;
plot(degrees,y);