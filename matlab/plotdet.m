function plotdet(llrs, labels, color)
%plots det curves for given system
maxaxis = 20;
thresholds = -70:0.1:70;

numtrue = ones(1, size(llrs, 1));%array with i-th element equal number of classes with that label

%set up labels as a logical matrix to make life easier
labelMat = true(size(llrs));
for i = 1:size(labelMat, 1)
    labelMat(i, :) = (labels == i);
    numtrue(i) = sum(labelMat(i, :));
end
faweight = 100./((size(labelMat, 2)-numtrue)*size(labelMat, 1));
frweight = 100./(numtrue*size(labelMat, 1));

y = ones(size(thresholds));
x = ones(size(thresholds));

for i = 1:size(thresholds, 2),
    dec = llrs > thresholds(i);
    y(i) = sum(frweight*(~dec & labelMat));
    x(i) = sum(faweight*(dec & ~labelMat));

    %fa = sum(dec(:) & ~labelMat(:));
    %fr = sum(~dec(:) & labelMat(:));
    %y(i) = fr*100.0/tot;
    %x(i) = fa*100.0/tot;
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