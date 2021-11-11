FILES = ["combined_data_1.txt", "combined_data_2.txt", "combined_data_3.txt", "combined_data_4.txt"]
OUTPUT_FILE = "consolidatedReviews.csv"
USER_DELIMITER = ":"
HEADING_LINE = "MovieID,UserID,Rating"

f = open(OUTPUT_FILE, 'w')
f.write(HEADING_LINE + "\n")


# We write to the file line by line
for fn in FILES:
    print("Opening...", fn)
    _file = open(fn)
    result = _file.readlines()

    currentItem = None
    for line in result:
        if USER_DELIMITER in line:
            currentItem = line[:line.index(USER_DELIMITER)]
            print("Current MovieID:", currentItem)  # Used to capture progress
        else:
            line = line.split(',')
            UserID = line[0]
            rating = line[1]
            rowStr = str(currentItem) + "," + str(UserID) + "," + str(rating)
            f.write(rowStr + "\n")
            #print(rowStr)
    print("Closing...", fn)
    _file.close()
f.close()
print("DONE!")