export MPJ_HOME=/Users/tianzhiliang/eclipse-workspace/MPIHello/mpj-v0_44/
export PATH=$MPJ_HOME/bin:$PATH

#javac -cp .:$MPJ_HOME/lib/mpj.jar Hello.java # method1 (it works)
#javac Hello.java -classpath $MPJ_HOME/lib/mpj.jar # method2 (it also works)

#javac -cp .:$MPJ_HOME/lib/mpj.jar MyHello.java # method1 (it works)
javac -cp .:$MPJ_HOME/lib/mpj.jar:$MPJ_HOME/lib/starter.jar MyHello.java # method1 (it works)
#javac MyHello.java -classpath $MPJ_HOME/lib/mpj.jar # method2 (it also works)
