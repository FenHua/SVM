%基于SVM实现特征分类
%kernel function
C=200;
ker=struct('type','linear');
%test sample
n=50;
randn('state',6);%保持现有状态
x1=randn(2,n);
y1=ones(1,n);
x2=5+randn(2,n);
y2=-ones(1,n);
figure(1);
plot(x1(1,:),x1(2,:),'bx',x2(1,:),x2(2,:),'k.');
axis([-3 8 -3 8]);
title('C-SVC')
hold on;
X=[x1,x2];
Y=[y1,y2];
%train SVM
tic;%tic用来保存当前的时间
svm=svmTrain('svc_c',X,Y,ker,C);
t_train=toc;%toc 记录程序完成的时间
%find sustain vector(支持向量)
a=svm.a;
epsilon=1e-8;
i_sv=find(abs(a)>epsilon);
plot(X(1,i_sv),X(2,i_sv),'ro');
%test
[x1,x2]=meshgrid(-2:0.1:7,-2:0.1:7);
[rows,cols]=size(x1);
nt=rows*cols;
Xt=[reshape(x1,1,nt);reshape(x2,1,nt)];
tic;
Yd=svmSim(svm,Xt);
t_sim=toc;
Yd=reshape(Yd,rows,cols);
contour(x1,x2,Yd,[0 0],'m');
hold off


















