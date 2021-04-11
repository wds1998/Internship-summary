fid = fopen ('source.data', 'rt');
source = polyshape.empty(5,0);
cnt = 1;
while feof(fid)~=1 
    base_str = fgetl(fid); 
    base = str2num(base_str);
    base_x =base(1:2:end);
    base_y =base(2:2:end);
    basepoly = polyshape(base_x, base_y);
    
    source(cnt) = basepoly;
    cnt=cnt+1;
end
fclose(fid);

fid = fopen ('test.data', 'rt');
target = polyshape.empty(5,0);
cnt = 1;
while feof(fid)~=1 
    base_str = fgetl(fid); 
    base = str2num(base_str);
    base_x =base(1:2:end);
    base_y =base(2:2:end);
    basepoly = polyshape(base_x, base_y);
    
    target(cnt) = basepoly;
    cnt=cnt+1;
end
fclose(fid);

k_num = 962;

global_td = turningdist(target, source');
[B, I] = mink(global_td, k_num, 1);

set(0,'DefaultFigureVisible', 'off')
[k_num, elem_num] = size(I);
for k = 1:20
   fprintf('%4.2f\n',global_td(k,1))

% for i = 1:elem_num
%     for k = 1: k_num
%         subplot(3,3,k);
%         plot(target(i))
%         hold on 
%         plot(source(I(k, i)))
%         title(sprintf('top%d, ≤Ó“Ï∂»£∫%4.2f',k, B(k,i)))
% 
%         axis equal
%         hold off       
%     end
%     saveas(gcf,sprintf('%d.jpg',i))
    %input('next')
end