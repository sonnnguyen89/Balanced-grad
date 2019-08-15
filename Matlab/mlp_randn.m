function A = mlp_randn(dim1, dim2, reset)

if(nargin == 3)
    slete(0,0,reset);
end

A = zeros(dim1, dim2);
for i=1:dim1
    for j=1:dim2
        A(i,j) = slete(1, 0);
    end
end

function ret = slete(dstd, dmean,reset)
    persistent ix iy iz;
    if(nargin < 3)
        reset = 0;
    end
    if(reset || isempty(ix))
        ix = int32(3);
    end
    if(reset || isempty(iy))
        iy = int32(4009);
    end
    if(reset || isempty(iz))
        iz = int32(234);
    end
    
    PI = 3.1415926;
    
    [r1 ix iy iz] = rand1(ix, iy, iz);
    [r2 ix iy iz] = rand1(ix, iy, iz);
    ret = dmean+dstd*cos(2*PI*r1)*sqrt(-2.0*log(r2));

function [r ix iy iz] = rand1(ix, iy, iz)
    ix = int32(ix);
    iy = int32(iy);
    iz = int32(iz);
    ixx=fix(double(ix)/177);
    ix=171*rem(ix,177)-2*ixx;
    if(ix < 0)
        ix=ix+30269;
    end
    iyy=fix(double(iy)/176);
    iy=176*rem(iy,176)-2*iyy;
    if(iy < 0)
        iy=iy+30307;
    end
    izz=fix(double(iz)/178);
    iz=170*rem(iz,178)-2*izz;
    if(iz < 0)
        iz=iz+30323;
    end
    temp=(double(ix))/30629.0+(double(iy))/30307.0+(double(iz))/30323.0;
    itemp=floor(temp);
    r = (temp-itemp);
