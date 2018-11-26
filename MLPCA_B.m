function [U,S,V,Ssq] = mlpca_b(X,Xsd,p);
%MLPCA_B Performs maximum likelihood principal components analysis for 
%        mode B error conditions (independent errors, homoscedastic 
%        within a column).  Equivalent to performing PCA on data 
%        scaled by the error SD, but results are rescaled to the original
%        space.
%
%   [U,S,V,SSQ,ERRFLAG] = MLPCA_C(X,XSD,P) produces the MLPCA estimate of
%   the rank P subspace for the IxJ matrix X, given the column measurement 
%   error standard deviations XSD (1xJ). Note that P must be less than the 
%   minimum of I and J.
%   
%   The returned parameters, U(IxP), S(PxP) and V(JxP), are analogs to the
%   truncated SVD solution (SVDS(X,P)), but have somewhat different 
%   properties since they represent the MLPCA solution.  In particular,
%   the eigenvectors do not necessarily account for decreasing 
%   amounts of variance, since MLPCA is a subspace modeling technique and
%   not a variance modeling technique.  The parameters returned are the
%   results of SVD on the estimated subspace.  The quantity SSQ represents
%   the sum of squares of weighted residuals.
%

% Section to check for valid inputs
%
if nargin~=3
    error('mlpca_b:err1','Incorrect number of arguments')
end
[m,n]=size(X);
p=p(1,1);              % In case of vector or matrix
if p>=min([m n])
    error('mlpca_b:err2','Invalid rank for MLPCA decomposition')
end
[m1,n1]=size(Xsd);
if (m1~=1 | n~=n1)
    error('mlpca_b:err3','Invalid dimension of SD vector')
end
Xsd=abs(Xsd);         % Ensure positive
indx=find(Xsd==0);
if length(indx~=0);
    error('mlpca_b:err4','Zero value(s)for standard deviations')
end
%
% Scale data, perform PCA, and then unscale
%
sclmat=ones(m,1)*Xsd;
Xsc=X./sclmat;
[U,S,V]=svds(Xsc,p);
Xcalc=(U*S*V').*sclmat;
[U,S,V]=svds(Xcalc,p);
Ssq=sum(sum(((X-Xcalc)./sclmat).^2));
