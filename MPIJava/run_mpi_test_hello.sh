export WORKDIR="/Users/tianzhiliang/Documents/Work/Code/LatentTree/hlta_parallel_0606/hlta_test_mpi/"
export CLASSPATH=${CLASSPATH}":"${WORKDIR}"/lib/mymedialite.jar:"${WORKDIR}"/lib/opencsv-3.6.jar"
export CLASSPATH=${CLASSPATH}":"${WORKDIR}"/lib/mpilib/mpj.jar:"${WORKDIR}"/lib/mpilib/starter.jar" 

#cd bin
java -jar ${WORKDIR}/lib/mpilib/starter.jar -np 4 Hello
#java -cp ${WORKDIR}/lib/mpilib/starter.jar Hello
#java -jar ${WORKDIR}/lib/mpilib/starter.jar -np 4 -classpath Hello
#cd -
