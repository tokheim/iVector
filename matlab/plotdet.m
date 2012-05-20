function plotdet(llrs, labels, color)
%plots det curves for given system
maxaxis = 2;
thresholds = -15:0.1:70;

tot = size(llrs, 2)*size(llrs, 1);
%set up labels as a logical matrix to make life easier
labelMat = logical(ones(size(llrs)));
for i = 1:size(labelMat, 1)
    labelMat(i, :) = (labels == i);
end
y = ones(size(thresholds));
x = ones(size(thresholds));

for i = 1:size(thresholds, 2),
    dec = llrs > thresholds(i);
    fa = sum(dec(:) & ~labelMat(:));
    fr = sum(~dec(:) & labelMat(:));
    y(i) = fr*100.0/tot;
    x(i) = fa*100.0/tot;
end

if nargin == 2,
    plot(x, y)
else
    plot(x, y, color)
end


%holdstatus = ishold;
%hold on
%plot(x, x, 'c')
%if ~holdstatus,
%    hold off
%end

axis([0 maxaxis 0 maxaxis])
xlabel('False alarm probability (in %)')
ylabel('Miss probability (in %)')
grid on