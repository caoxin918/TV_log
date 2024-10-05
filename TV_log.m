function [x_TV]=TV_log(nodesMat,y,A1,tau,err)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 计算TV_log
% %% imput %%%%%%%%%%
% y: the length-K observation vector 
%    (y=Ax, x is the original sparse signal),
% A: the measurement matrix with size KxN,
% tau: the regularization parameter,0.0005
% %% output %%%%%%%%%%%
% x_tv: the solution of min 1/2||Ax-y||_2^2+tau log (TV(X))
%       computed by TV_log, it is an approximation of x,
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_nodes=size(nodesMat,1);
%求体表节点的索引
nodes_y=zeros(size(y,1),1);
h=1;
for i=1:num_nodes
    if nodesMat(i,4)~=0
        nodes_y(h,1)=i;
        h=h+1;
    end
end
%计算每个点到其余点的欧式距离
Z=pdist(nodesMat(:,1:3));
mindistance=squareform(Z);
for i=1:num_nodes
    mindistance(i,i)=Inf;
end
%找到最近像素值的索引
[~,min_index]=min(mindistance,[],2);
D = zeros(num_nodes,num_nodes);
%计算差分矩阵
n=0;
for i=1:num_nodes
    z=min_index(i,1);
    if D(z,i)==1 
        continue;
    end
    D(i,z)=1;
    n=n+1;
    D(i,i)=-1;
end
for i=1:num_nodes
    a=find(D(:,i)==1);
    if size(a,1)==0||size(a,1)==1
        continue;
    end
    for j=2:size(a,1)        
            D(a(j,1),:)=0;
            n=n-1;
    end   
end 
DX=zeros(n,num_nodes);
n=1;
for i=1:num_nodes
    if D(i,:)~=0 
        DX(n,:)=D(i,:);
        n=n+1;
    end
end
disp(size(DX));
%%初始化
%重叠组尺寸
K=3;
conv_h=ones(1,K);
% x的初始值
x=pinv(A1)*y;
x(nodes_y)=0;
iter=0; 
t_max=10;
ErrR=norm(y-A1*x);
while (ErrR<=err)||(iter<=t_max)   
    C=1./sqrt(conv(transpose(abs(DX*x).^2),conv_h,'same'));
    D1=transpose(DX)*diag((C).^2)*DX;
    A=transpose(A1)*A1+tau*D1;
    b=transpose(A1)*y;
    %%PCG
    %w:超松弛因子
    w=0.01;                                                                                                               ;
    B=diag(diag(A));
    B1=sqrt(B/w);
    L=triu(A)-B;
    M=transpose(pinv(B1)*(B1.^2+L))*(pinv(B1)*(B1.^2+L))/(1*(2-w));%计算预处理矩阵M  
    r=b-A*x;
    invm=pinv(M);
    z=invm*r;
    p=z;
    k=1;
   while (k<10)  %max(abs(Error))>1e-10
     zr=z'*r; 
     alpha=zr/(p'*(A*p));
     x=x+alpha*p;
     x(nodes_y)=0; 
     r=r-alpha*A*p;
     z=invm*r;
     beta=(z'*r)/zr;
     p=z+beta*p;
     ErrR=norm(y-A1*x);
     k=k+1;
   end   
   x(nodes_y)=0;
   NewX=x;
   iter=iter+1;
end
x_TV=NewX;


    


