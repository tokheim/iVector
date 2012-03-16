function threeclass_scatterplot(data,classf);
f1=find(classf==1);
f2=find(classf==2);
f3=find(classf==3);

if size(data,1)==2
   d2 = data;
elseif size(data,1)==3
   d2 = [1,0,-1; 0,1,-1]*data;
else
   error('illegal argument data, must be 2- or 3-dimensional');
end;
plot(d2(1,f1),d2(2,f1),'.r',d2(1,f2),d2(2,f2),'.g',d2(1,f3),d2(2,f3),'.b');