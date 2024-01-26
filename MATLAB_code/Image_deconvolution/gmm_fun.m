function y = gmm_fun(x)

% define GMM (prior) f(x)
%eps = 0.005;
eps = 0.15;
s0 = 0.0002;
s1 = 10;
s = 0.04;
r = 0.9;
m0 = 0;
m1 = 0;

C0 = @(x) (1/sqrt(2.*pi.*(eps^2 + s0^2)).* exp(-(x-m0).^2./(2.*(eps^2 + s0^2))));

C1 = @(x) (1/sqrt(2.*pi.*(eps^2 + s1^2)).* exp(-(x-m1).^2./(2.*(eps^2 + s1^2))));

C = @(x) r .* C0(x) + (1-r) .* C1(x);

w_tilde = @(x) (C0(x).*r)./(C(x));

d0_sqr = (eps^2.*s0^2)./(eps^2 + s0^2);
d1_sqr = (eps^2.*s1^2)./(eps^2 + s1^2);

mu0 = @(x) (x./eps^2 + m0./s0^2).* d0_sqr;
mu1 = @(x) (x./eps^2 + m1./s1^2).* d1_sqr;

% GMM: D(x) - x
f = @(x) w_tilde(x) .* mu0(x) + (1-w_tilde(x)) .* mu1(x) - x;

y = f(x);

end