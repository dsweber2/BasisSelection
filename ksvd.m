function [minThis, dict, x] = ksvd(data, k, maxIter, peps, pMaxIter)
    epss=1e-6; dictInit=@randoDict; pursuitAlgorithm=@simpleBasisPursuit;
    % initialization
    [n m] = size(data);
    dict = randCols(data, k);%dictInit(k, n);

    x = zeros(k, m);
    i=0;
    minThis = [Inf];

    x = pursuitAlgorithm(dict, data, x, peps, pMaxIter, f);
    while i<maxIter && minThis(end) >= epss
        for jj=1:k
            ind = jj; % indexUpdate(E);

            v = find(x(ind,:));
            if nnz(v) > 0
                Q= zeros(m,m); ii=0;
                for l=1:length(v)
                    Q(v(l),v(l)) = 1;
                end
                [U S V] = svd(E*Q);

                dict(:, ind) = U(:,1);
                tmp = S(1,1)*Q*V;
                x(ind, :) = tmp(:,1)'; % matters to calculate error
                minThis(end+1) = norm(data-dict*x);
            else
                indNeglect = 0;
                valNeglect = 0;
                for ll=1:m
                    temp = norm(data(:,ll)-dict*x(:,ll),2);
                    if temp >= valNeglect
                        indNeglect = ll;
                        valNeglect = temp;
                    end
                end
                if indNeglect==0
                    indNeglect = randi(m);
                end
                dict(:,ind) = data(:,indNeglect)/norm(data(:,indNeglect));
                x(ind,:) = zeros(1,m);
                x(ind,indNeglect) = 1/norm(data(:,indNeglect));
            end
        end
        i=i+1
        minThis(end)
        x = pursuitAlgorithm(dict, data, x, peps, pMaxIter, f);
    end
end

function x=simpleBasisPursuit(dict, data, x, epss, maxIter, f)
    t = now();
    [n m] = size(data);
    [k m] = size(x);
    for i=1:m
        lambda = 1/2^3;
        [temp inform] = as_bpdn(dict, data(:,i), lambda);
        while (nnz(temp)>n*epss) && (lambda < 2^10)
            lambda = 2*lambda;
            [temp inform] = as_bpdn(dict, data(:,i), lambda);
        end
        x(:,i)=temp;
    end
end


function out=randoDict(k, n)
    out = rand(n, k);
    for i = 1:k
        out(:,i) = out(:,i)./norm(out(:,i));
    end
end


function dict=randCols(data, k)
% choose a random assortment of data columns to be
    [n m] = size(data);
    dict = data(:,randperm(m,k));
    for i=1:k
        dict(:,i) = dict(:,i)/norm(dict(:,i));
    end
end
