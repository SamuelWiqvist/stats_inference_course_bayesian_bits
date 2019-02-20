
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
