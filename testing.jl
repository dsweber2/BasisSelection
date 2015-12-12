Pkg.add(Blas)
for y in [1,2,3,4]
  println(y)
end

dict = [[1/sqrt(2) 1/sqrt(2) 0 1/sqrt(3)], [0 1/sqrt(2) 0 1/sqrt(3)], [1/sqrt(2) 0 1/sqrt(2) 1/sqrt(3)]]
dict = [[1 1 0 1],[0 1 0 1], [1 0 1 1]]
data = [[1. 1. 0],[0 1. 0]]'
n=3; m=2; k=4
y=[1.,0,3.];m=1;
x=[0.,0,0,0]

dict*x
r = copy(y)
# loop starts
for i=1:100000
  diff=dict'*r # the ijth entry is the inner product between the ith data vector and the jth dictonary element
  ind = [indmax(abs(diff[:,w])) for w=1:m]
  maxdif = [diff[ind[w],w] for w=1:m]
  for w=1:m
    x[ind[w],w] += maxdif[w]
    r[:,w]-= dict[:,ind[w]]*x[ind[w],w]
  end
end
y
y-dict*x
r
dict*x
data
# loop ends

x =[[0. 0 0 0],[0. 0 0. 0]]'
r = copy(data)
# loop starts
diff=dict'*r # the ijth entry is the inner product between the ith data vector and the jth dictonary element
diff
ind = [indmax(abs(diff[:,w])) for w=1:m]
maxdif = [diff[ind[w],w] for w=1:m]
for w=1:m
  x[ind[w],w] = maxdif[w]
  r[:,w]-= dict[:,ind[w]]*x[ind[w],w]
end
x
r[:,1]'*dict[:,[ind[1]]]
r[:,2]'*dict[:,[ind[2]]]

data-dict*x
dict*x
data
# loop ends

x

i


zeros(1,2)

out=rand(3,20)

out[:,3]./norm(out[:,3])

for i=1:n
end

using PyPlot
plot(ree,imm)
ree = -3./(3^2+t.^2)
imm = -t./(3^2+t.^2)

t=[-.01:-.01:-1,-1:-100,-100:-10:-10000,.01:.01:1,1:100,100:10:10000]
t=[-10000:-1, -1:-.001:0, 0:.001:1, 1:100000]
rey = real(1./(t*im+3)./(1./(t*im+3)+1))
imy = imag(1./(t*im+3)./(1./(t*im+3)+1))
plot(rey,imy)
PyPlot.show()


plot(re(1/(im*ω-3)),)

using Convex

m=4;n=5;

A = randn(m,n)
b= rand(m,1);

x=Variable(n)

problem = minimize(norm(A*x-b,1),[x>=0])

solve!(problem)

problem.status

problem.optval

problem.solution.dual

dict = [[1 1 0 1],[0 1 0 1], [1 0 1 1]]
x =[[0. 0 0 0],[0. 0 0. 0]]'
[norm(x[:,i]) for i=1:2]
data = [[1. 0 1.],[1 1. 0]]'
x=Variable(4,2)
n=3; m=2; k=4
sparsify = minimize(vecnorm(data-dict*x))
for i=1:2
  sparsify.constraints += norm(x[:,i],1)<=1
end
sparsify.constraints
solve!(sparsify)
x.value
x
sparsify.status
sparsify.solution.primal
dict*x.value

solve!()
w=time()
sparsify.solution.primal
w-time()
x.value
a
f = open("name.txt","a")
t=time(); peps = 3; pMaxIter = 100; pursuitAlgorithm = simpleBasisPursuit;
write(f, join(["\n\nStarting a new iteration at ",t, " with pursuit error ",
        peps," and maximum iterations ", pMaxIter, " using pursuit algorithm ",
        dump(pursuitAlgorithm), ]))
write(f,a)
write(f,"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
write(f, join(["\nStarting a new iteration at ",time()]))
write(f, "Here's some words and other stuff")
writedlm(f,dict)
close(f)

:simpleBasisPursuit

dump(simpleBasisPursuit)
sparsify.optval

rwet =[1 44 3 4 5; 6 7 20 9 -8; 11 12 -24 14 12]

norm(rwet[:,w])
data = round(randn(5,30)) #rand(9,84)
k = 14
sum([norm(data[i,:] - dict[i,:]*x.value,2) for i=1:5])
include("ksvd.jl"); (minn,dict,x)= ksvd(data, k, 2, 5, 10, 1)
dict*x.value
vcat([zeros(1,4) for i=1:2], [1, 2, 3, 4]', zeros(1,4))
zeros(1,4)
[zeros(1,4) for i=1:2]
dict[:,3]
x.value[3,:]
x.value = randn(7,15)
x.value
ind = indmax([norm(rwet[:,w]) for w=1:5])
srand
diagm
x=Variable(7,15)
x.value
v = 3
10e-5
function wae()
  return 3
end

wae.name
show(wae)
ceil([10e-6,10e-3]-10e-5)
epss=10e-5
71^70-70^71
x.value
diagm(vec(ceil(x.value[ind,:]-epss)))
ceil(x.value[ind,:]-epss)
if pursuitAlgorithm!=simpleBasisPursuit

  t=1
else
  t=3
end

# dictionary update
ind = 1; n=3; m=5; k=4
function randoDict(k, n)
    # create k normalized column vectors of length 1
    out = rand(n, k)
    for i = 1:k
        out[:,i] = out[:,i]./norm(out[:,i])
    end
  return out
end
dict = randoDict(k,n)
norm(data-dict*x.value)
#dict = [[1. 1. 0 1],[0 1 0 1], [1 0 1 1]]
data = [[1. 0 1.],[1 1. 0],[1 0 0],[0 0 1], [0 -1 1]]'
x=Variable(4,5)
sparsify = minimize(vecnorm(data-dict*x))
sparsify
for i=1:5
  sparsify.constraints += norm(x[:,i],1)<=2
end
x
solve!(sparsify)
sparsify.optval
x.value
E = data - dict*x.value
ind = 4 # indexUpdate(E)
v = ceil(abs(x.value[ind,:])-epss)
w = nnz(v)
Q= zeros(m,w); ii=0
for l=1:m
  if v[l]!=0
    ii+=1
    Q[l,ii] =1
  end
end

F = svdfact(E*Q)
dict[:,1] = F[:U][:,1]
x.value[1,:] = (F[:S][1]*Q*F[:V])[:,1]

norm(data-dict*x.value)
dict[:,1]
join(dict[:,1],", ")

