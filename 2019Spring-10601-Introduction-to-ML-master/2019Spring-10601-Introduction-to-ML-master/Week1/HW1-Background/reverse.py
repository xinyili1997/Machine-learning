from __future__ import print_function
import sys

# define the main() function for reverse operation with three sub-functions:
# readFile(), reverseFile() and writeFile()
def main():
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    f_read = readFile(in_file)
    f_reverse = reverseFile(f_read)
    writeFile(out_file, f_reverse)
    return


# define the readFile() function to read lines from in_file
def readFile(in_file):
    f = open(in_file, "r")
    line_list = f.readlines()
    return line_list
# readFile operation returns a list of lines of in_file


# define the reverseFile() function to reverse the line_list with for loop
def reverseFile(line_list):
    reverse_list = []
    for i in range(len(line_list)):
        reverse_list.append(line_list[-1-i])
    return reverse_list
# reverseFile operation returns a list of reversed lines of in_file


# define the writeFile() function to write the reverse_list with for loop
def writeFile(out_file, reverse_list):
    f = open(out_file, "w")
    for i in range(len(reverse_list)):
        f.write(reverse_list[i])
    return
# output with out_file



if __name__ == '__main__':
    main()