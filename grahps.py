import csv
import matplotlib.pyplot as plt
import numpy as np

def extract_numbers(file_path, start_row, end_row):
    numbers_list = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            if start_row <= idx <= end_row:
                close_value = row['Close/Last']
                # Extracting the number after the $ symbol and converting it to a float
                number = float(close_value.split('$')[1])
                numbers_list.append(number)
    return numbers_list

# Replace 'Data.csv' with the path to your CSV file
file_path = 'Data.csv'

# Example: Extracting numbers from row 1 to row 5
start_row = 2000
end_row = 2099

N = extract_numbers(file_path, start_row, end_row)
N.reverse()




def read_floats_from_file(file_path):
    numbers_list = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                number = float(line.strip())  # Convert each line to a float
                numbers_list.append(number)
            except ValueError:
                print(f"Skipped line: {line.strip()} - Not a valid float")
    return numbers_list

# Replace 'results.txt' with the path to your actual text file
file_path = 'results.txt'

R = read_floats_from_file(file_path)


mse = np.mean((np.array(N) - np.array(R))**2)
print("Mean Squared Error (MSE):", mse)

mae = np.mean(np.abs(np.array(N) - np.array(R)))
print("Mean Absolute Error (MAE):", mae)



plt.figure(figsize=(10, 10))  # Set the size of the plot

plt.plot(N, label='N')  # Plotting list N
plt.plot(R, label='R')  # Plotting list R

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Comparison of N and R')

plt.legend()  # Displaying legend
plt.grid(True)  # Adding gridlines

plt.show()  # Display the plot

