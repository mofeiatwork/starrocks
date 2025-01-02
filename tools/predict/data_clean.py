import re
import csv
import sys

# Define the parsing function
def parse_line(line):
    """
    Parse a single line of data, extract cpuCostNs, memCostBytes and features.
    """
    match = re.search(r'cpuCostNs=(\d+)\|memCostBytes=(\d+)\|features=([\d,]+)', line)
    if match:
        cpu_cost = int(match.group(1))
        mem_cost = int(match.group(2))
        # Rule out the zero cpu_cost or mem_cost
        if cpu_cost == 0 or mem_cost == 0:
            return None
        features = list(map(int, match.group(3).split(',')))
        return [cpu_cost, mem_cost] + features
    return None

def process_file(input_file, output_file):
    """
    Process the input file and write the parsing result to a CSV file.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        header_written = False
        
        for line in infile:
            parsed = parse_line(line.strip())
            if parsed:
                # Write the header
                if not header_written:
                    header = ['cpuCostNs', 'memCostBytes'] + [f'feature_{i}' for i in range(len(parsed) - 2)]
                    writer.writerow(header)
                    header_written = True
                # Write the data
                writer.writerow(parsed)

# Main function
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_clean.py <input_file> <output_file>")
        sys.exit()
    input_file = sys.argv[1]  # Read the input file name from the command line
    output_file = sys.argv[2]  # Read the output file name from the command line
    
    print(f"Processing file: {input_file}")
    process_file(input_file, output_file)
    print(f"Data has been written to: {output_file}")