function [U,S,V,Ssq,ErrFlag] = mlpca_c(X,Xsd,p);
%MLPCA_C Performs maximum likelihood principal components analysis for 
%        mode C error conditions (independent errors, general heteroscedastic 
%        case).  Employs ALS algorithm.
%
%   [U,S,V,SSQ,ERRFLAG] = MLPCA_C(X,XSD,P) produces the MLPCA estimate of
%   the rank P subspace for the IxJ matrix X, given the measurement error 
%   standard deviations XSD (IxJ). Note that P must be less than the minimum 
%   of I and J. For missing values in X, the XSD values should be set to
%   NaN.
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
%   the sum of squares of weighted residuals. ERRFLAG indicates the 
%   convergence condition, with 0 indicating normal termination and 1
%   indicating the maximum number of iterations have been exceeded.
%

% Section to check for valid inputs
%
if nargin~=3
    error('mlpca_c:err1','Incorrect number of arguments')
end
[m,n]=size(X);
p=p(1,1);              % In case of vector or matrix
if p>=min([m n])
    error('mlpca_c:err2','Invalid rank for MLPCA decomposition')
end
[m1,n1]=size(Xsd);
if (m~=m1 | n~=n1)
    error('mlpca_c:err3','Dimensions of data and standard deviations do not match')
end
Xsd=abs(Xsd);         % Ensure positive
indx=find(Xsd==0);
if length(indx~=0);
    error('mlpca_c:err4','Zero value(s)for standard deviations')
end
%
% Initialization
%
convlim=1e-10;             % convergence limit
maxiter=2000;              % maximum no. of iterations
varmult=1000;              % multiplier for missing data
varX=(Xsd.^2);             % convert s.d.'s to variances
indx=find(isnan(varX));    % find missing values
varmax = max(max(varX));   % maximum variance
varX(indx)=varmax*varmult; % give missing values large variance
%
% Generate initial estimates assuming homoscedastic errors.
%
[U,S,V]=svds(X,p);         % Decompose adjusted matrix
count=0;                   % Loop counter
Sold=0;                    % Holds last value of objective function
ErrFlag=-1;                % Loop flag
%
% Loop for alternating regression
%
while ErrFlag<0;
   count=count+1;          % Increment loop counter
%   disp(count)
%
% Evaluate objective function
%
   Sobj=0;                             % Initialize sum      
   MLX=zeros(size(X));                 % and maximum likelihood estimates
   for i=1:n                           % Loop for each column of XX
      Q=diag(1./varX(:,i));            % Inverse of error covariance matrix
      F=inv(U'*Q*U);                   % Intermediate calculation
      MLX(:,i)=U*(F*(U'*(Q*X(:,i))));  % Max. likelihood estimates
      dx=X(:,i)-MLX(:,i);              % Residual vector
      Sobj=Sobj+dx'*Q*dx;              % Update objective function
   end
%
% This section for diagnostics only and can be commented out.  "Ssave"
% can be plotted to follow convergence.
%
%   Ssave(count)=Sobj;
%   save mlpca;
%
% End diagnostics
%
% Check for convergence or excessive iterations
%
   if rem(count,2)==1                   % Check on odd iterations only
      if (abs(Sold-Sobj)/Sobj)<convlim  % Convergence criterion
         ErrFlag=0;
      elseif count>maxiter              % Excessive iterations?
         ErrFlag=1;
         warning('mlpca_c:err5','Maximum iterations exceeded')
      end
   end
%
% Now flip matrices for alternating regression
%
   if ErrFlag<0                         % Only do this part if not done
      Sold=Sobj;                        % Save most recent obj. function
      [U,S,V]=svds(MLX,p);              % Decompose ML values
      X=X';                             % Flip matrix
      varX=varX';                       % and the variances
      n=length(X(1,:));                 % Adjust no. of columns
      U=V;                              % V becomes U in for transpose
   end
end
%
% All done.  Clean up and go home.
%
[U,S,V]=svds(MLX,p);
Ssq=Sobj;
