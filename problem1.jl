# load packages
using Optim
using LinearAlgebra
using Distributions
using ForwardDiff
using KernelDensity
using PyPlot
using Random

# fix random numbers
Random.seed!(1234)

# load file
file = open("eBayNumberOfBidderData.dat")
data_file = readlines(file)
close(file)

# load convaraites and targets
nbr_obs = 1000
nbr_covariates = 9
X = zeros(nbr_covariates,nbr_obs)
y = zeros(nbr_obs)

for i = 1:nbr_obs
    file_line = split(data_file[i+1])
    for j = 1:nbr_covariates
        X[j,i] = parse(Float64,file_line[j+1])
    end
    y[i] = parse(Float64,file_line[1])
end

# set prior
τ_0 = 10
prior_cov_m = τ_0^2*Matrix{Float64}(I, nbr_covariates, nbr_covariates)
prior_cov_m_inv = inv(prior_cov_m)
dist_marginal_prior = Normal(0, τ_0)

# optmization problem

# logposterior function
function logposterior(β)

    logpos = 0.

    for i = 1:nbr_obs

        logpos = logpos + ((X[:,i]'*β)*y[i] - exp(X[:,i]'*β))

    end

    logpos = logpos - 0.5*β'*prior_cov_m_inv*β

    return -logpos # Optim does minimization, hence the minus sign

end

# start values for optimizer
β_start = ones(nbr_covariates)

# run optimization using conjugate gradient decent with numerical gradients
opt = optimize(logposterior, β_start, ConjugateGradient())

# get parameter estimations (posterior mean)
β_tilde = Optim.minimizer(opt)

numerical_hessian = Symmetric(ForwardDiff.hessian(logposterior, β_tilde))

# sample from approx posterior
posterior_mean = β_tilde

posterior_cov_m = inv(-numerical_hessian) + 10^(-2)*Matrix{Float64}(I, nbr_covariates, nbr_covariates)

approx_posterior = MvNormal(posterior_mean, posterior_cov_m)

posterior_samples = rand(approx_posterior, 10^4)


# plot results

h1_β1 = kde(posterior_samples[1,:])
h1_β2 = kde(posterior_samples[2,:])
h1_β3 = kde(posterior_samples[3,:])
h1_β4 = kde(posterior_samples[4,:])
h1_β5 = kde(posterior_samples[5,:])
h1_β6 = kde(posterior_samples[6,:])
h1_β7 = kde(posterior_samples[7,:])
h1_β8 = kde(posterior_samples[8,:])
h1_β9 = kde(posterior_samples[9,:])

println("Posterior (marginal) means:")
println(round.(mean(posterior_samples,dims = 2); digits=3))

println("Posterior (marginal) std:")
println(round.(std(posterior_samples,dims = 2); digits=3))


text_size = 10
PyPlot.figure(figsize=(12,12))
PyPlot.subplot(331)
PyPlot.plot(h1_β1.x,h1_β1.density, "b")
PyPlot.plot(h1_β1.x,pdf.(dist_marginal_prior, h1_β1.x), "g")
PyPlot.xlabel(L"\beta_1",fontsize=text_size)
PyPlot.subplot(332)
PyPlot.plot(h1_β2.x,h1_β2.density, "b")
PyPlot.plot(h1_β2.x,pdf.(dist_marginal_prior, h1_β2.x), "g")
PyPlot.xlabel(L"\beta_2",fontsize=text_size)
PyPlot.subplot(333)
PyPlot.plot(h1_β3.x,h1_β3.density, "b")
PyPlot.plot(h1_β3.x,pdf.(dist_marginal_prior, h1_β3.x), "g")
PyPlot.xlabel(L"\beta_3",fontsize=text_size)
PyPlot.subplot(334)
PyPlot.plot(h1_β4.x,h1_β4.density, "b")
PyPlot.plot(h1_β4.x,pdf.(dist_marginal_prior, h1_β4.x), "g")
PyPlot.xlabel(L"\beta_4",fontsize=text_size)
PyPlot.subplot(335)
PyPlot.plot(h1_β5.x,h1_β5.density, "b")
PyPlot.plot(h1_β5.x,pdf.(dist_marginal_prior, h1_β5.x), "g")
PyPlot.xlabel(L"\beta_5",fontsize=text_size)
PyPlot.subplot(336)
PyPlot.plot(h1_β6.x,h1_β6.density, "b")
PyPlot.plot(h1_β6.x,pdf.(dist_marginal_prior, h1_β6.x), "g")
PyPlot.xlabel(L"\beta_6",fontsize=text_size)
PyPlot.subplot(337)
PyPlot.plot(h1_β7.x,h1_β7.density, "b")
PyPlot.plot(h1_β7.x,pdf.(dist_marginal_prior, h1_β7.x), "g")
PyPlot.xlabel(L"\beta_7",fontsize=text_size)
PyPlot.subplot(338)
PyPlot.plot(h1_β8.x,h1_β8.density, "b")
PyPlot.plot(h1_β8.x,pdf.(dist_marginal_prior, h1_β8.x), "g")
PyPlot.xlabel(L"\beta_8",fontsize=text_size)
PyPlot.subplot(339)
PyPlot.plot(h1_β9.x,h1_β9.density, "b")
PyPlot.plot(h1_β9.x,pdf.(dist_marginal_prior, h1_β9.x), "g")
PyPlot.xlabel(L"\beta_9",fontsize=text_size)


# simulate from posterior predictive

posterior_samples = rand(approx_posterior, 10^3)
posterior_pred_bids = zeros(10^3)

x_case = [1, 1, 1, 1, 0, 0, 0, 1, 0.5]

for i = 1:10^3
    sample_bids = Poisson(exp(x_case'*posterior_samples[:,i]))
    posterior_pred_bids[i] = rand(sample_bids)
end

println("Posterior predictive mean:")
println(mean(posterior_pred_bids))

println("Posterior predictive std:")
println(std(posterior_pred_bids))


max_bids = Int(maximum(posterior_pred_bids))
nbr_bids = zeros(max_bids+1)

for i = 0:max_bids
    nbr_bids[i+1] = length(findall(x->x==i, posterior_pred_bids))
end


PyPlot.figure(figsize=(6,6))
b = bar(0:max_bids,nbr_bids/10^3)
PyPlot.xlabel("Bids",fontsize=text_size)
PyPlot.ylabel("Freq.",fontsize=text_size)
