#!/bin/bash

# On this date all posts were updated with the date field
# We want to ignore this date
all_change_date=2020-05-12

# Go through all files in the wiki folder
for file in `find wiki/ -name '*.md'`; do
	# Get last date on which file was updated 
	last_change=$(git log -1 --format="%ad" --date=short $file)
	if [ "$last_change" = "$all_change_date" ]; then
		# If the last change was on the all_change_date
		# set last_change to last change before all_change_date
		last_change=$(git log -1 --format="%ad" --before=$all_change_date --date=short $file)
	fi
	# Replace date filed with last_change
	sed -i "s/date:.*/date: $last_change/" $file;
done
