# TODO:
#   figure out how to choose the upper limit on the 1-norm
#   normalize the index selection
#   index over Q rather than making new matrices
using Convex
using SCS
function ksvd(data, k, maxIter, peps, pMaxIter, pursuitAlgorithm=1, epss=1e-3,
    dictInit=randoDict)
    # pursuitAlgorithm: is the handle of one of the functions listed below; 
    #   they have arguments (dict, data, x, eps, maxIter),
    # data: a set of column vectors representing whatever we're trying to construct 
    #   a dictionary for.
    # peps: the error tolerance for the pursuit algorithm; for now, it represents 
    #   the fraction of nonzero entries in x
    # pMaxIter: the maximum number of iterations we'll pursue a basis
    # deps: the error tolerance for the dictionary
    # k: the size of the dictionary

    # seeding the RNG; this also belongs in a macro
    # srand(1)

    # logs of performance; this really belongs in a macro
    f = open("C:\\Users\\Owner\\Documents\\GitHub\\BasisSelection\\logs.txt","a")
    t = time()
    write(f,"\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    write(f,"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    write(f,"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    write(f, join(["\n\nStarting a new run at ",t, " with pursuit error ",
        peps," and maximum iterations ", pMaxIter, " using pursuit algorithm ",
        dump(pursuitAlgorithm), " looking for a dictionary of size ", k,"\n"]))

    # initialization
    (n,m) = size(data)
    condNum = vecnorm(data)/m/n # not actually a condition number
    dict = dictInit(k, n)   # the dictionary we're training, column vectors initialized
                            # through some method. Default is to randomly generate 
                            # vectors in [0,1]^n and normalize

    x = Variable(k,m)    # the sparse representation; should be column vectors;
                        # each of them has the same dimension as a dictionary vector

    perforated = []     # a record of the performance
    i=0
    minThis = Inf

    # selects the Algorithm
    if pursuitAlgorithm==1
        pursuitAlgorithm = simpleBasisPursuit
    elseif pursuitAlgorithm==2
        pursuitAlgorithm = coordinateBP
        x.value = zeros(k,m)
    elseif  pursuitAlgorithm==3
        pursuitAlgorithm = switchBasisPursuit
    elseif pursuitAlgorithm==4
        pursuitAlgorithm = logBasisPursuit
    end

    x = pursuitAlgorithm(dict, data, condNum, x, peps, pMaxIter, f) # if I want
    # to use
            # other pursuit algorithms, I'll need to update this
    write(f, "\nThe initial sparse vector is\n")
    #writedlm(f, x.value)
    write(f,"\n")
    while i<maxIter && minThis >= epss
        # sparse coding

        # dictionary update
        # update across all indices
        for j=1:k
            relRow = zeros(n,m)
            for l=1:n
                for o=1:m
                    relRow[l,o] = dict[l,j]*x.value[j,o]
                end
            end
            # write(f,"\n and now, the row we're replacing:\n")
            # writedlm(f, relRow)
            E = data - dict*x.value + relRow
            # write(f,"\n the error in everything else\n")
            # writedlm(f, E)
            ind = j # indexUpdate(E)
            v = ceil(abs(x.value[ind,:])-epss)
            w = nnz(v)
            if w>0
                Q= zeros(m,w); ii=0
                # println(w)
                for l=1:m
                    if v[l]!=0
                        ii+=1
                        Q[l,ii] = 1
                    end
                end
                # write(f, "\n Q, or the rows that are non-zero\n")
                # minimize the error for the single index index using the SVD
                F = svdfact(E*Q)

                # for reasons I'm not sure about, the objective function isn't
                # strictly decreasing; I think that's the primary problem I need
                # to solve
                # write(f,"\nSVD time\n U=")
                # writedlm(f,F[:U])
                # write(f, "\nD=")
                # writedlm(f,F[:S])
                # write(f, "\nV=")
                # writedlm(f, F[:V])
                dict[:, ind] = F[:U][:,1]
                # write(f,"P")
                x.value[ind, :] = (F[:S][1]*Q*F[:V])[:,1]
                minThis = norm(data-dict*x.value)
                write(f, join(["\n changed basis element ", j, " and got a",
                            " global objective of ", vecnorm(data-dict*x.value)]))
            else
                # replace our unused dictionary element with a poorly
                # represented data vector
                indNeglect = 0
                valNeglect = 0
                for ll=1:m
                    temp = norm(data[:,ll]-dict*x.value[:,ll],2)
                    if temp >= valNeglect
                        indNeglect = ll
                        valNeglect = temp
                    end
                end
                # Apparently we hate all the data equally
                if indNeglect==0
                    indNeglect = rand(1:m)
                    valNeglect = norm(data[:,indNeglect])
                end
                dict[:,ind] = data[:,indNeglect]/norm(data[:,indNeglect])
                # Nothing to be done but sparsify again, I suppose
                x = pursuitAlgorithm(dict, data, x, peps, pMaxIter, f)
                # x.value[ind,:] = zeros(1,m)
                # x.value[ind,indNeglect] = norm(data[:,indNeglect])
                write(f,join(["\nthis basis vector isn't being used, substituted ",
                    "the ", indNeglect, "th sample\n"]))
            end
        end
        i+=1
        write(f,"\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        write(f,join(["\nJust finished the ", i, "th iteration of dictionary ",
            "updates; at the moment, the dictionary is\n"]))
        #writedlm(f,dict)
        write(f,"\nwhile the sparsity vector x is\n")
        #writedlm(f,x.value)
        x = pursuitAlgorithm(dict, data, x, peps, pMaxIter, f)
    end
    write(f, join(["\nWell boys, it's been a good run; we've had ", i, " runs, ",
        "taking a total of ", time()-t, " seconds, with final estimation ",
        "matrix\n"]))
    #writedlm(f, dict*x.value)
    close(f)
    return (minThis, dict, x)
end

function simpleBasisPursuit(dict, data, condNum, x, epss, maxIter, f)
    # use the methods of Convex to run basis pursuit coordinate-wise
    # write(f,"INTO THE BREACH")
    # write(f,"0")
    t = time()
    (n,m) = size(data)
    # write(f,"1")
    sparsify = minimize(vecnorm(data - dict*x))
    # write(f,"2")
    # iterate over each vector in the data
    for i=1:m
        sparsify.constraints += norm(x[:,i],1)<=epss*condNum
    end
    # write(f,"3")
    solve!(sparsify)
    write(f, join(["\nwe've sparsified again!\n Status:", sparsify.status, 
        "       error: ", sparsify.optval,"       Time:", time()-t]))
    return x
end

function logBasisPursuit(dict, data, x, epss, maxIter, f)
    # use the methods of Convex to run basis pursuit coordinate-wise
    # write(f,"INTO THE BREACH")
    # write(f,"0")
    t = time()
    (n,m) = size(data)
    # write(f,"1")
    sparsify = minimize(vecnorm(data - dict*x))
    # write(f,"2")
    # iterate over each vector in the data
    for i=1:m
        sparsify.constraints += sum([log((x[j,i])^2+1) for j=1:n])<=epss*m
    end
    # write(f,"3")
    solve!(sparsify, SCS.SCSSolver(max_iters=maxIter))
    write(f, join(["\nwe've sparsified again!\n Status:", sparsify.status, 
        "       error: ", sparsify.optval,"       Time:", time()-t]))
    return x
end

function switchBasisPursuit(dict, data, x, epss, maxIter, f)
    # use the methods of Convex to run basis pursuit coordinate-wise
    # write(f,"INTO THE BREACH")
    # write(f,"0")
    t = time()
    (n,m) = size(data)
    # write(f,"1")
    sparsify = minimize(sum([norm(x[:,i],1) for i=1:m]))
    # write(f,"2")
    sparsify.constraints= sum([norm(data[i,:] - dict[i,:]*x,2) for i=1:m]) <=
    epss*m
    # write(f,"3")
    solve!(sparsify)
    write(f, join(["\nwe've sparsified again!\n Status:", sparsify.status, 
        "       error: ", sparsify.optval,"       Time:", time()-t]))
    return x
end

# do BP for each constraint separately
function coordinateBP(dict, data, x, epss, maxIter, f)
    # use the methods of Convex to run basis pursuit coordinate-wise
    # write(f,"INTO THE BREACH")
    # write(f,"0")
    t = time()
    (n,m) = size(data)
    # write(f,"1")
    # write(f,"2")
    # iterate over each vector in the data
    for i=1:m
        y = Variable(k)
        sparsify = minimize(vecnorm(data[:,i] - dict*y))
        sparsify.constraints += norm(y,1)<=epss
        solve!(sparsify)
        x.value[:,i] = y.value
    end
    # write(f,"3")
    write(f, join(["\nwe've sparsified again!\n Status:", "sparsify.status", 
        "       error: ", "sparsify.optval","       Time:", time()-t]))
    return x
end




function matchingPursuit(dict, data, x, epss, maxIter,f)
    # the simplest algorithm
    i=1
    while i<=maxIter && norm(r) > epss
        diff=dict'*r    # the ijth entry is the inner product between the ith data 
                        # vector and the jth dictonary element
        ind = [indmax(abs(diff[:,w])) for w=1:m]
        maxdif = [diff[ind[w],w] for w=1:m]
        for w=1:m
            x[ind[w],w] += maxdif[w]
            r[:,w]-= dict[:,ind[w]]*x[ind[w],w]
        end
        i+=1
    end
    return x
end

function randoDict(k, n)
    # create k normalized column vectors of length 1
    out = rand(n, k)
    for i = 1:k
        out[:,i] = out[:,i]./norm(out[:,i])
    end
    return out
end

# this issue of order may be useful later, but for now it's unnecessary
function indexUpdate(E)
    # we need to choose which index to change; for now I'm going to use the one
    # with the worst square error
    ind = indmax([norm(E[:,w]) for w=1:5])
    return ind
end


function sparsity(x)
# given a vector x, calculate the sparsity (with error 1e-3)
    v = ceil(abs(x)-1e-3)
    (n,m)=size(v)
    function temp(x)
        if x>0
            return 1
        else
            return 0
        end
    end
    indicate = [[temp(v[i,j]) for i=1:n] for j=1:m]
    return sum(sum(indicate))/n/m
end
