import os, subprocess


output = subprocess.check_output("earthengine ls projects/cipalawan/assets/alertsv4",shell=True)

myList = output.decode('utf8').split("\n")

counter = 0
for items in myList:
	try:
		myName = items.split("/")[4]
		year = myName[17:21]
		month = myName[21:23]
		day = myName[23:25]
		#print(items, counter, year, month, day)

		time = str(year) + "-" + str(month) + "-" + str(day) + "T00:00:00"
		counter+=1
		os.system("earthengine asset set --time_start " + time + " " + items)
	except:
		pass
