  /*
   * Author of revised version: Franklyn Pinedo
   * Author of new revised version: David Walker
   *
   * Adapted from Source Code in C of Tutorial/User's Guide for MPI by
   * Peter Pacheco.
   */
  import java.util.*;

  class HelloWorld {
    static public void main(String[] args) {
      StringBuilder sb = new StringBuilder(); 
      Map<String, String> env = System.getenv(); 
      System.out.println("will print");
      System.out.println("len of keySet:" + env.keySet().size());
      for (String key : env.keySet()) { 
          sb.append(key + ": " + env.get(key)  + "\n"); 
      } 

// now the StringBuilder sb contains  all the enviroment variables, and can be logged or displayed to the servlet or whatever
//
      System.out.println(sb.toString());
      System.out.println("Done");
    }
  }
