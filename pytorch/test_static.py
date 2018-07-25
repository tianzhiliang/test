def myfunc():
    myfunc.counter += 1
    print(myfunc.counter)

myfunc.counter = 0
myfunc()
myfunc()
myfunc()

