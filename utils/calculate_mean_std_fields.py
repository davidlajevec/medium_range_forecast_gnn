import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

def calculate_mean_std_fields_parallel(root, atmosphere_variables, start_year, end_year):
    # Read annotations once
    with open(f"{root}/annotation_file.txt", "r") as f:
        annotations = f.readlines()

    # Prepare file names
    file_names = {variable: [] for variable in atmosphere_variables}
    years = set(range(start_year, end_year + 1))

    for line in annotations:
        year = int(line[2:6])
        if year in years:
            for variable in atmosphere_variables:
                file_names[variable].append(
                    f"{variable}/{variable}_{year}/{variable}{line[1:-1]}"
                )
    if end_year == 2022:
        for variable in atmosphere_variables:
            file_names[variable].pop(-1)

    # Define a function to process each variable
    def process_variable(variable):
        mean = np.zeros((60, 120))
        std = np.zeros((60, 120))

        data_list = []
        for file_name in file_names[variable]:
            data = np.load(f"{root}/{file_name}")
            data_list.append(data)
            mean += data
        mean /= len(file_names[variable])

        for data in data_list:
            std += (data - mean) ** 2
        std /= len(file_names[variable])
        std = np.sqrt(std)

        # Save the results
        os.makedirs(f"data_12hr/{variable}", exist_ok=True)
        np.save(f"data_12hr/{variable}/{variable}_mean.npy", mean)
        np.save(f"data_12hr/{variable}/{variable}_std.npy", std)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=len(atmosphere_variables)) as executor:
        executor.map(process_variable, atmosphere_variables)

# Call the function
# calculate_mean_std_fields_parallel(root, atmosphere_variables, start_year, end_year)
if __name__ == "__main__":
    calculate_mean_std_fields_parallel("data_12hr", 
        ["geopotential_250",
        "geopotential_500",
        "geopotential_700",
        "geopotential_850",
        "mslp",
        "potential_vorticity_500",
        "potential_vorticity_300",
        "precipitation",
        "sic",
        "snow",
        "soil_moisture_layer_1",
        "surface_temperature",
        "t2m",
        "t500",
        "t800",
        "toa_solar_radiation",
        "twv",
        "u_10m",
        "v_10m",
        "u_200",
        "v_200",
        "u_500",
        "v_500",
        "u_700",
        "v_700",
        "u_900",
        "v_900",
         ],
        1950, 
        2008
        )