import math
import csv
import sys


def inspect_data(csv_file):
    """
    Input: .csv file
    1. read the .csv, ignore the heading and gather the label information in labels{}
    2. counting the binary labels and gather in labels_binary{}
    3. call function cal_entropy_margin and cal_error_rate
    :return entropy_margin, error_rate
    """
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        labels = []
        for row in reader:
            labels.append(row[-1])
    labels_binary = {}

    for label in labels:
        if label in labels_binary:
            labels_binary[label] += 1
        else:
            labels_binary[label] = 1

    entropy_margin = cal_entropy_margin(labels_binary.values())
    error_rate = cal_error_rate(labels_binary.values())

    return entropy_margin, error_rate


def cal_entropy_margin(labels):
    """
    Input: labels_binary list with counters of {+, -}
    def function cal_entropy_margin
    :return: entropy_margin
    """
    entropy_margin = 0
    for label in labels:
        entropy_margin += -(label / sum(labels)) * math.log2(label / sum(labels))
    return entropy_margin


def cal_error_rate(labels):
    """
    Input: labels_binary list with counters of {+, -}
    def function cal_error_rate
    :return: error_rate
    """
    return 1 - max(labels) / sum(labels)


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    entropy_margin, error_rate = inspect_data(input_file)
    outfile = open(output_file, 'w')
    outfile.write("entropy: {}\n".format(entropy_margin))
    outfile.write("error: {}\n".format(error_rate))
    output_file.close()


if __name__ == "__main__":
    main()
