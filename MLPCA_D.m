function [U,S,V,Ssq]=mlpca_d(X,Cov,p)
%MLPCA_D Performs maximum likelihood principal components analysis for 
%        mode D error conditions (common row covariance matrices). 
%        Employs rotation and scaling of the original data.
%
%   [U,S,V,SSQ] = MLPCA_D(X,COV,P) produces the MLPCA estimate of the
%   rank P subspace for the IxJ matrix X, given the measurement error 
%   covariance matrix COV (JxJ), which is common to all of the rows.   
%   
%   The returned parameters, U(IxP), S(PxP) and V(JxP), are analogs to the
%   truncated SVD solution (SVDS(X,P)), but have somewhat different 
%   properties since they represent the MLPCA solution.  In particular, 
%   the solutions for different values of P are not necessarily nested
%   (the rank 1 solution may not be in the space of the rank 2 solution)
%   and the eigenvectors do not necessarily account for decreasing 
%   amounts of variance, since MLPCA is a subspace modeling technique and
%   not a variance modeling technique.  The parameters returned are the
%   results of SVD on the estimated subspace.  The quantity SSQ represents
%   the sum of squares of weighted residuals (objective function).
%

[m,n]=size(X);
df=(m-p)*(n-p);
[U1,S1,V1]=svd(Cov);
covrank=rank(Cov);      % If the covariance matrix is singular, then
if covrank<n            % expand uniformly in deficient directions
   scale=[sqrt(diag(S1(1:covrank,1:covrank)))'...
       ones(1,n-covrank)*sqrt(S1(covrank,covrank))*0.01];
else
   scale=sqrt(diag(S1));
end    
Z=X*U1*diag(1./scale);
[U2,S2,V2]=svd(Z);
Zcalc=U2(:,1:p)*S2(1:p,1:p)*V2(:,1:p)';
Ssq=0;
for i=1:m;
   Ssq=Ssq+norm(Zcalc(i,1:covrank)-Z(i,1:covrank))^2;
end
Xcalc=Zcalc*diag(scale)*U1';
[U,S,V]=svd(Xcalc);
U=U(:,1:p);
S=S(1:p,1:p);
V=V(:,1:p);

