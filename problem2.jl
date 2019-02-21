# load packages
using Optim
using Distributions
using KernelDensity
using PyPlot
using LinearAlgebra
using Random

# fix random numbers
Random.seed!(1234)

# load file
file = open("bivarnormal.dat")
data_file = readlines(file)
close(file)

# load data
nbr_obs = 20
dimensions = 2
X = zeros(dimensions,nbr_obs)


for i = 1:nbr_obs
    file_line = split(data_file[i+1])
    for j = 1:dimensions
        if file_line[j] == "NA"
            X[j,i] = NaN
        else
            X[j,i] = parse(Float64,file_line[j])
        end
    end
end

# remove NaNs TODO: Do something for the missing values!
X_data = zeros(2,nbr_obs-3)
X_data[:,1:2] = X[:,1:2]
X_data[:,3:end] = X[:,6:end]
X = X_data
nbr_obs = nbr_obs-3

# prior dist
Σ_0 = 10^2*Matrix{Float64}(I, dimensions, dimensions)
μ_0 = zeros(dimensions)

dist_marginal_prior = Normal(0, 10)


# data model
Σ = [1 1.5; 1.5 4]

# Joint posterior (see https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution)
x_bar = mean(X, dims = 2) # sample mean
μ_post = inv((inv(Σ_0)+nbr_obs*inv(Σ)))*(inv(Σ_0)*μ_0 + nbr_obs*inv(Σ)*x_bar)
Σ_post = inv((inv(Σ_0)+nbr_obs*inv(Σ)))

# Gibbs sampler
function gibbs(N_samples, μ_post = μ_post, Σ_post = Σ_post)

    # construct conditional dists
    μ_post_1 = μ_post[1]
    μ_post_2 = μ_post[2]

    Σ_post_11 = Σ_post[1,1]
    Σ_post_22 = Σ_post[2,2]
    Σ_post_12 = Σ_post[1,2]

    # pre-allocate matrix
    μ_post_sample = zeros(dimensions, N_samples)

    # set start value
    μ_2_old = 100

    for i = 1:N_samples

        # sample 1 give 2
        post_cond_1_give_2_m = μ_post_1+Σ_post_12*inv(Σ_post_22)*(μ_2_old-μ_post_2)
        post_cond_1_give_2_std = sqrt(Σ_post_11 - Σ_post_12*inv(Σ_post_22)*Σ_post_12)
        sample_cond_1_give_2 = rand(Normal(post_cond_1_give_2_m, post_cond_1_give_2_std))

        # sample 2 give 1
        post_cond_2_give_1_m = μ_post_2+Σ_post_12*inv(Σ_post_11)*(sample_cond_1_give_2-μ_post_1)
        post_cond_2_give_1_std = sqrt(Σ_post_22 - Σ_post_12*inv(Σ_post_11)*Σ_post_12)
        sample_cond_2_give_1 = rand(Normal(post_cond_2_give_1_m, post_cond_2_give_1_std))

        # store samples
        μ_post_sample[:,i] = [sample_cond_1_give_2;sample_cond_2_give_1]

        # update 2 give 1
        μ_2_old = sample_cond_2_give_1

    end

    return μ_post_sample

end

# run gibbs sampler
nbr_samples = 10^3
burn_in = 100
post_samples = gibbs(nbr_samples+burn_in)

# plotting

# entier chain
PyPlot.figure()
PyPlot.subplot(211)
PyPlot.plot(post_samples[1,:], "b")
PyPlot.subplot(212)
PyPlot.plot(post_samples[2,:], "b")

# chain after burnin
PyPlot.figure()
PyPlot.subplot(211)
PyPlot.plot(post_samples[1,burn_in+1:end], "b")
PyPlot.subplot(212)
PyPlot.plot(post_samples[2,burn_in+1:end], "b")

h1_μ_1 = kde(post_samples[1,burn_in+1:end])
h1_μ_2 = kde(post_samples[2,burn_in+1:end])

println("Posterior (marginal) means:")
println(mean(post_samples[:,burn_in+1:end],dims = 2))

println("Posterior (marginal) std:")
println(std(post_samples[:,burn_in+1:end],dims = 2))

# marginal posteriors
PyPlot.figure()
PyPlot.subplot(211)
PyPlot.plot(h1_μ_1.x,h1_μ_1.density, "b")
PyPlot.plot(h1_μ_1.x,pdf.(dist_marginal_prior, h1_μ_1.x), "g")
PyPlot.subplot(212)
PyPlot.plot(h1_μ_2.x,h1_μ_2.density, "b")
PyPlot.plot(h1_μ_2.x,pdf.(dist_marginal_prior, h1_μ_2.x), "g")

# find optimal action numerically

# utility function
function utility(α, μ)

    ϕ = min(μ[1], μ[2])

    return -max(0, ϕ-α)-0.1*ϕ*α^2

end

# expected utility function
function expected_utility(α, posterior_samples)

    N = size(posterior_samples,2)
    u_vec = zeros(N)

    for i = 1:N
        u_vec[i] = utility(α, posterior_samples[:,i])
    end


    return mean(u_vec)

end

obj_func(α) = -expected_utility(α[1], post_samples[:,burn_in+1:end]) # Optim does minimization, hence the minus sign

α_start = [10.]

opt = optimize(obj_func, α_start, BFGS())

α_tilde = Optim.minimizer(opt)

println("Optimal action:")
println(α_tilde)
