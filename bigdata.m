close all
clear all
srcFiles = dir('C:\Users\HaoNo\Downloads\jaffeimages\jaffe\*.tiff');  % the folder in which ur images exists
M = length(srcFiles);
T = zeros(65536,M);
for i = 1 : length(srcFiles)
    filename = strcat('C:\Users\HaoNo\Downloads\jaffeimages\jaffe\',srcFiles(i).name);
    I = imread(filename);
   T(:,i) = reshape(I,[65536,1]);
end

Psi = 1/M*sum(T,2);%the average face
Fi = T-repmat(Psi,[1,M]);
A = Fi;
figure
imshow(reshape(Psi(:,1),[256,256]),[])
%show the average face

C = A'*A;
[V,E] = eig(C);%Calculate eigenfaces and eigenvalues
eigval = diag(E);
V = fliplr(V);
K = 20;%The top 20 vector
U = zeros(65536,K);
Y = Psi;
W = zeros(K,1);
figure
% The last several eigenfaces
for i=205:212
    U(:,i) = A*V(:,i);
    U(:,i) = U(:,i)/sum(U(:,i),1);
   subplot(3,4,i)
   imshow(reshape(U(:,i),[256,256]),[])
end

testPath = 'C:\Users\HaoNo\Documents\MATLAB\ee.bmp';
test_image = imread(testPath);
% test_image = rgb2gray(test_image);
test_image = im2double(test_image);
test_image = imresize(test_image,[256,256]);
test_image = reshape(test_image,[65536,1]);
%test_image = T(:,23);
Omega = U'*(test_image-Psi);
Dist = zeros(M,1);
for i=1:1:M
    Omega_i = U'*Fi(:,i);
    dist = norm(Omega-Omega_i);
    Dist(i) = dist;
end
[val,idx] = min(Dist);
e = 100;
if val<e
    fprintf('The face is recognized as face %i\n\n\n\n',idx)
else
    fprintf('There is no face matched\n')
end