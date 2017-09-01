function [Q,R] = calculateHQR (A)
% NOT WORKING
% http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
	n = size(A);
	Q = eye(n);
	R = A;
	for j = 1:n
		normx = norm(R(j:end,j));
		s = -sign(R(j,j));
		u1 = R(j,j) - s*normx;
		w = R(j:end,j)/u1;
		w(1) = 1;
		tau = -s*u1/normx;
		R(j:end,:) = R(j:end,:)-(tau*w)*(transpose(w)*R(j:end,:));
		Q(:,j:end) = Q(:,j:end)-(Q(:,j:end)*w)*(tau*w)';
	end
end

function H = calculateHessenberg(A)
	[m,n] = size(A);
	L = zeros(m,n);
	H = A;
	for j = 1:m-2
		x = H(j+1:m,j);
		x(1) = x(1) + sign(x(1)) * norm(x);
		n = norm(x);
		if n > 0
			u = x/norm(x);
			H(j+1:m,j:m) = H(j+1:m,j:m) - 2*u*(transpose(u)*H(j+1:m,j:m));
			H(1:m,j+1:m) = H(1:m,j+1:m) - 2*( H(1:m,j+1:m)*u )*u';
		else
			u = x;
		end
        %L(j+1:m,j) = u;
        end
end

function [Q, R] = calculateQR(A)

  n = rows(A);
  R = A;
  Q = eye(n);
  for i = 1:n
    for j = i+1:n
      if R(j,i) == 0
        c = 1;
        s = 0;
      elseif abs(R(j,i)) < abs(R(i,i))
        t = R(j,i) / R(i,i);
        c = 1 / sqrt(1 + t^2);
        s = c*t;
      else
        z = R(i,i) / R(j,i);
        s = 1 / sqrt(1 + z^2);
        c = s*z;
      endif
      G = [c,s;-1*s,c];
      R([i,j],i:n) = G * R([i,j],i:n);
      Q([i,j],1:n) = G * Q([i,j],1:n);
    end
  end
  Q = Q';

end

function E = eig2p2 (A)
	p = [ 1 , -A(1,1)-A(2,2) , A(1,1)*A(2,2) - A(1,2)*A(2,1) ];
	E = roots(p);
end

function ANS = calculateEigenvalues(A)

  tic;
  if (rows(A) == columns(A))
    A = hessenberg(A);
    ANS = transpose(calculateDSQR(A));
  else
    printf ("Error: The given matrix is not squared.\n");
  endif
  toc;
end

function ANS = calculateDSQR(A)
% http://web.stanford.edu/class/cme335/lecture5

  max_iterations = 50;
  convergence = 0.0001;
  eigenvalues = [];
  rows = rows(A);

  do
    for i = 1:max_iterations
      [Q,R] = calculateQR(A);
      A = R*Q;
      if abs(A(rows, rows-1)) < convergence
        eigenvalues(end+1) = A(rows, rows);
        A = A(1:rows-1, 1:rows-1);
        rows = rows - 1;
        break;
      end

      if i == max_iterations
        submatrix = A(rows-1:rows, rows-1:rows);
        b = -1 * (submatrix(1, 1) + submatrix(2, 2));
        c = (submatrix(1, 1) * submatrix(2, 2)) - (submatrix(1, 2) * submatrix(2, 1));

        determ = b^2 - 4*c;
        if determ >= 0
          eigenvalues(end+1) = (-1*b + sqrt(determ)) / 2;
          eigenvalues(end+1) = (-1*b - sqrt(determ)) / 2;
        end

        if determ < 0
          eigenvalues(end+1) = complex(-1*b/2, sqrt(determ*-1) / 2);
          eigenvalues(end+1) = complex(-1*b/2, -1*sqrt(determ*-1) / 2);
        end

        A = A(1:rows-2, 1:rows-2);
        rows = rows - 2;
      end
    end
  until rows < 2

  eigenvalues(end+1) = A;
  ANS = eigenvalues;

end
