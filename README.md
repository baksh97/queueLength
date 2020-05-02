# queueLength
To run the code, type the following commands:
  1. make
  2. ./code_no_dyn.o <video_name> 0 0 0 0
  
Some things you might need to change:
  1. In the makefile, you may need to change opencv4 to opencv
  2. The background.png is an image taken out of the video which shows just the empty road. It is needed to get the values of static density. So change it according to the video but keep the name as "background.png". This image is taken for the video "Approach3_Back_2019_11_23_03_58_08-Small-Day.mp4"
  3. If you are running this for any other video, run using the command "./code_no_dyn.o <video_name> 1 0 0 0" for the first time, to generate the txt file as explained by shivam and vedant in their code. After first run and choosing the points, you can run with the command provided in the starting.
  4. Inside the code, MAXFPS_LIM has been hardcoded to 5. You may change it to any other value by setting the second last value argument to any other value other than 0 and less than the video fps.
  5. The value of alpha1 (STATIC_FACTOR) has been hardcoded too. You may not need to change it as this value works just fine for static average.
  6. If you dont need to see the output using imshow, just comment out the lines from 1091- 1100

Points to note:
  1. Results will come in a file named: "results_gpu_final_<STATIC_FACTOR>.txt"
  2. Results file has: <Result counter>: <Static/Queue Density> <Stop Density> <Queue Length>
  3. We only need the first 3 columns for our current analysis.
  4. The result is printed to the file every second. Every 5th second, Sachin's code runs and hence we dont get the values of those seconds. This may be changed inside the code at line number 888.
