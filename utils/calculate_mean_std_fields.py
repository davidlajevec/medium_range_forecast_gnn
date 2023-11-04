import numpy as np

def calculate_mean_std_fields(root, atmosphere_variables, start_year, end_year):    
    with open(f"{root}/annotation_file.txt", "r") as f:
        annotations = f.readlines()
    
    file_names = [[] for variable in atmosphere_variables]
    
    years = list(range(start_year, end_year + 1))
    for line in annotations:
        year = int(line[2:6])
        if year in years:
            for i, variable in enumerate(atmosphere_variables):
                file_names[i].append(
                    f"{variable}/{variable}_{year}/{variable}{line[1:-1]}"
                )
    if end_year == 2022:
        for i, variable in enumerate(atmosphere_variables):
            file_names[i].pop(-1)
    
    mean = np.zeros((len(atmosphere_variables), 60, 120))
    std = np.zeros((len(atmosphere_variables), 60, 120))
    for i, variable in enumerate(atmosphere_variables):
        for file_name in file_names[i]:
            data = np.load(f"{root}/{file_name}")
            mean[i] += data
        mean[i] /= len(file_names[i])
        for file_name in file_names[i]:
            data = np.load(f"{root}/{file_name}")
            std[i] += (data - mean[i]) ** 2
        std[i] /= len(file_names[i])
        std[i] = np.sqrt(std[i])
    return mean, std

if __name__ == "__main__":
    mean, std = calculate_mean_std_fields("data_12hr", ["geopotential_500", "u_500", "v_500"], 1950, 2008)
    
    geopotential_500_mean = mean[0]
    geopotential_500_std = std[0]

    u_500_mean = mean[1]
    u_500_std = std[1]

    v_500_mean = mean[2]
    v_500_std = std[2]

    np.save("data_12hr/geopotential_500/geopotential_500_mean.npy", geopotential_500_mean)
    np.save("data_12hr/geopotential_500/geopotential_500_std.npy", geopotential_500_std)

    np.save("data_12hr/u_500/u_500_mean.npy", u_500_mean)
    np.save("data_12hr/u_500/u_500_std.npy", u_500_std)

    np.save("data_12hr/v_500/v_500_mean.npy", v_500_mean)
    np.save("data_12hr/v_500/v_500_std.npy", v_500_std)

