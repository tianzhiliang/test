export WORKDIR="/Users/tianzhiliang/Documents/Work/Code/LatentTree/hlta_parallel_0606/hlta_test_mpi/"
export CLASSPATH=${CLASSPATH}":"${WORKDIR}"/lib/mymedialite.jar:"${WORKDIR}"/lib/opencsv-3.6.jar"
export CLASSPATH=${CLASSPATH}":"${WORKDIR}"/lib/mpilib/mpj.jar:"${WORKDIR}"/lib/mpilib/starter.jar" 


#java -cp "HLTA_without_mpi.jar:"${WORKDIR}"/lib/mymedialite.jar:"${WORKDIR}"/lib/opencsv-3.6.jar":${WORKDIR}"/lib/mpilib/starter.jar" runStepwiseEMHLTA   # can run. this is the way to use multiple jar
java -jar ${WORKDIR}"/lib/mpilib/starter.jar" -cp ${WORKDIR}"/target/scala-2.12/HLTA.jar:"${WORKDIR}"/lib/mymedialite.jar:"${WORKDIR}"/lib/opencsv-3.6.jar" -np 4 runStepwiseEMHLTA # can run. this is the way to run strater which depandency(external memory or from code)
