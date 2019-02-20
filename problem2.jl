# load packages 
using Optim

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


# how to structe the optimization problem
#using Optim
#rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
#result = optimize(rosenbrock, zeros(2), BFGS())

# optmization problem

# logposterior function
function logposterior(β)




end

# start values for optimizer
β_start = ones(nbr_covariates)

# run optimization using conjugate gradient decent with numerical gradients
opt = optimize(logposterior, β_start, ConjugateGradient())

# get parameter estimations (posterior mean)
β_tilde = Optim.minimizer(opt)

# get Hessian matrix
numerical_hessian = hessian!(logposterior,β_tilde)

# sample from approx posterior


# plot results
