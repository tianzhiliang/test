#$ -S /bin/bash

# environment
export MPJ_HOME=/bigstore/hlcm2/tianzhiliang/latentTree/javampi/mpj-v0_44
# specify where the output file should be put
# option
#this_is_output_path

# specify the working path
#this_is_working_path

# email me with this address...
#$ -M tianzhiliang@ust.hk
# email when the job starts (b) and after the job has been 
# completed (e)
#$ -m be

cd bin
/usr/local/packages/j2se-9/bin/java -jar ${MPJ_HOME}/lib/starter.jar -np 1 HelloWorld 
cd -
