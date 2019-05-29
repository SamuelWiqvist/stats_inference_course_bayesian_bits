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



# prior dist
Σ_0 = Symmetric(10^2*Matrix{Float64}(I, dimensions, dimensions))
μ_0 = zeros(dimensions)

dist_marginal_prior = Normal(0, 10)

# data model
Σ = Symmetric([1 1.5; 1.5 4])

# Gibbs sampler
function gibbs(N_samples, Σ_0 = Σ_0, Σ=Σ, X_data = X, dimensions=dimensions)


    # pre-allocate matrix
    missing_data_values = zeros(3, N_samples)
    μ_post_sample = zeros(dimensions, N_samples)

    # set start value
    μ_1_old = 100
    μ_2_old = 100

    for i = 1:N_samples

        # sample missing data values
        mean_missing_1 = μ_1_old + Σ[1,2]*inv(Σ[2,2])*(X[2,3]-μ_2_old)
        mean_missing_2 = μ_1_old + Σ[1,2]*inv(Σ[2,2])*(X[2,4]-μ_2_old)
        mean_missing_3 = μ_2_old + Σ[2,1]*inv(Σ[1,1])*(X[1,5]-μ_1_old)

        sigma2_missing_1 = Σ[1,1]-Σ[1,2]*inv(Σ[2,2])*Σ[2,1]
        sigma2_missing_2 = Σ[1,2]-Σ[1,2]*inv(Σ[2,2])*Σ[2,1]
        sigma2_missing_3 = Σ[2,2]-Σ[2,1]*inv(Σ[1,1])*Σ[1,2]


        missing_data_values[1,i] = mean_missing_1 + sqrt(sigma2_missing_1)*randn()
        missing_data_values[2,i] = mean_missing_2 + sqrt(sigma2_missing_2)*randn()
        missing_data_values[3,i] = mean_missing_3 + sqrt(sigma2_missing_3)*randn()

        X_data[1,3] = missing_data_values[1,i]
        X_data[1,4] = missing_data_values[2,i]
        X_data[2,5] = missing_data_values[3,i]

        # Joint posterior (see https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution)
        x_bar = mean(X_data, dims = 2) # sample mean
        μ_post = inv((inv(Σ_0)+nbr_obs*inv(Σ)))*(inv(Σ_0)*μ_0 + nbr_obs*inv(Σ)*x_bar)
        Σ_post = inv((inv(Σ_0)+nbr_obs*inv(Σ)))

        println(μ_post)
        println(Σ_post)

        post =  MvNormal(μ_post[:], Σ_post)

        # store samples
        μ_post_sample[:,i] = rand(post)

        # μ_old
        μ_1_old,μ_2_old = μ_post_sample[:,i]

    end

    return μ_post_sample, missing_data_values

end

# run gibbs sampler
nbr_samples = 10^3
burn_in = 100
μ_post_sample,missing_data_values = gibbs(nbr_samples+burn_in)

# plotting
text_size = 10

# entier chain
PyPlot.figure(figsize=(10,6))
PyPlot.subplot(311)
PyPlot.plot(missing_data_values[1,:], "b")
PyPlot.ylabel("Missing value 1",fontsize=text_size)
PyPlot.subplot(312)
PyPlot.plot(missing_data_values[2,:], "b")
PyPlot.ylabel("Missing value 2",fontsize=text_size)
PyPlot.subplot(313)
PyPlot.plot(missing_data_values[3,:], "b")
PyPlot.xlabel(L"Iteration",fontsize=text_size)
PyPlot.ylabel("Missing value 3",fontsize=text_size)

PyPlot.figure(figsize=(10,6))
PyPlot.subplot(311)
PyPlot.plot(missing_data_values[1,burn_in+1:end], "b")
PyPlot.ylabel("Missing value 1",fontsize=text_size)
PyPlot.subplot(312)
PyPlot.plot(missing_data_values[2,burn_in+1:end], "b")
PyPlot.ylabel("Missing value 2",fontsize=text_size)
PyPlot.subplot(313)
PyPlot.plot(missing_data_values[3,burn_in+1:end], "b")
PyPlot.xlabel(L"Iteration",fontsize=text_size)
PyPlot.ylabel("Missing value 3",fontsize=text_size)



# chain after burnin
PyPlot.figure(figsize=(10,6))
PyPlot.subplot(211)
PyPlot.plot(μ_post_sample[1,burn_in+1:end], "b")
PyPlot.ylabel(L"\mu_1",fontsize=text_size)
PyPlot.subplot(212)
PyPlot.plot(μ_post_sample[2,burn_in+1:end], "b")
PyPlot.xlabel(L"Iteration",fontsize=text_size)
PyPlot.ylabel(L"\mu_2",fontsize=text_size)

# entier chain
PyPlot.figure(figsize=(10,6))
PyPlot.subplot(211)
PyPlot.plot(μ_post_sample[1,:], "b")
PyPlot.ylabel(L"\mu_1",fontsize=text_size)
PyPlot.subplot(212)
PyPlot.plot(μ_post_sample[2,:], "b")
PyPlot.xlabel(L"Iteration",fontsize=text_size)
PyPlot.ylabel(L"\mu_2",fontsize=text_size)

h1_μ_1 = kde(μ_post_sample[1,burn_in+1:end])
h1_μ_2 = kde(μ_post_sample[2,burn_in+1:end])

println("Posterior (marginal) means:")
println(round.(mean(μ_post_sample[:,burn_in+1:end],dims = 2); digits=3))

println("Posterior (marginal) std:")
println(round.(std(μ_post_sample[:,burn_in+1:end],dims = 2); digits=3))

# marginal posteriors
PyPlot.figure(figsize=(10,6))
PyPlot.subplot(121)
PyPlot.plot(h1_μ_1.x,h1_μ_1.density, "b")
PyPlot.plot(h1_μ_1.x,pdf.(dist_marginal_prior, h1_μ_1.x), "g")
PyPlot.xlabel(L"\mu_1",fontsize=text_size)
PyPlot.subplot(122)
PyPlot.plot(h1_μ_2.x,h1_μ_2.density, "b")
PyPlot.plot(h1_μ_2.x,pdf.(dist_marginal_prior, h1_μ_2.x), "g")
PyPlot.xlabel(L"\mu_1",fontsize=text_size)

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
println(round.(α_tilde; digits=3))
