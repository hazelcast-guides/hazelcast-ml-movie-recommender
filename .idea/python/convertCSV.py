import re

# reading given tsv file
with open("tsv_file", 'r') as myfile:
    with open("csv_file", 'w') as csv_file:
        for line in myfile:

            # Replace every tab with comma
            fileContent = re.sub("\t", ",", line)

            # Writing into csv file
            csv_file.write(fileContent)

# output
print("Successfully made csv file")