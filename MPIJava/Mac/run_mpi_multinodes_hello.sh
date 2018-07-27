#export WORKDIR="/Users/tianzhiliang/Documents/Work/Code/LatentTree/hlta_parallel_0606/hlta_test_mpi/"
#export CLASSPATH=${CLASSPATH}":"${WORKDIR}"/lib/mymedialite.jar:"${WORKDIR}"/lib/opencsv-3.6.jar"
#export CLASSPATH=${CLASSPATH}":"${WORKDIR}"/lib/mpilib/mpj.jar:"${WORKDIR}"/lib/mpilib/starter.jar" 
export WORKDIR="/Users/tianzhiliang/eclipse-workspace/MPIHello/mpj-v0_44/"
export MPJ_HOME="/Users/tianzhiliang/eclipse-workspace/MPIHello/mpj-v0_44/"
export PATH=$MPJ_HOME/bin:$PATH
export CLASSPATH=${CLASSPATH}":"${MPJ_HOME}"/lib/mpj.jar:"${MPJ_HOME}"/lib/starter.jar" 

#sh mpjrun.sh -np 4 -dev native Hello
sh mpjrun.sh -np 4 -dev hybdev Hello
