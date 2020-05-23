# queueLength
To run the code, type the following commands:
  1. make
  2. ./code_no_dyn.o <video_name_without_extension> 0 0 0 0
  
Some things you might need to change:
  1. If the video extension is other than mp4, change it in the code in file code_no_dyn.cpp at line 671 ("video_format").
  2. In the makefile, you may need to change opencv4 to opencv
  3. Code needs 3 files to run which are named as, (a) bg_<video_name_without_extension>.png, (b) traffic_<video_name_without_extension>.png, (c) <video_name_without_extension>.txt. (a) represents a background image from the video without traffic, (b) represents an image from the video with complete traffic and (c) represents the projected points from video. If you want to run the code on the sample files, rename "Approach3_Back_2019_11_23_03_58_08-Small-Day.mp4" as a3b_day.mp4
  4. If you are running this for any other video, run using the command "./code_no_dyn.o <video_name_without_extension> 1 0 0 0" for the first time, to generate the txt file as explained by shivam and vedant in their code. After first run and choosing the points, you can run with the command provided in the starting.
  5. Inside the code, MAXFPS_LIM has been hardcoded to 5. You may change it to any other value by setting the second last value argument to any other value other than 0 and less than the video fps.
  6. The value of alpha1 (STATIC_FACTOR) has been hardcoded too. You may not need to change it as this value works just fine for static average.
  7. If you dont need to see the output using imshow, just comment out the lines from 1137- 1147

Points to note:
  1. Results will come in a file named: "results_<video_name_without_extension>.txt"
  2. Results file has: <Result counter>: <Static/Queue Density> <Stop Density>
  4. The result is printed to the file every second. Every 5th second, Sachin's code runs and hence we dont get the values of those seconds. This may be changed inside the code at line number 916.
