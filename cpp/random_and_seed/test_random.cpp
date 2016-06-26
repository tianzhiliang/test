#include <stdio.h>
#include <stdlib.h>
#include <ctime> // or #include <time.h>
#include <sys/time.h>

void set_m_sec_seed() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    unsigned int time = static_cast<unsigned int>(tv.tv_sec * 1000 * 1000 + tv.tv_usec);
    srand((unsigned int)time);
    printf("time:%u\n", time);
}

void run_random(int size, float range) {
    for (int i = 0; i < size; i++) {
        printf("i:%d rand:%d\n", i, rand());
    }
    printf("\n");
    
    for (int i = 0; i < size; i++) {
        printf("i:%d rand:%f\n", i, rand() * 1.0 / RAND_MAX);
    }
    printf("\n");

    for (int i = 0; i < size; i++) {
        printf("i:%d rand:%f\n", i, rand() * 1.0 / RAND_MAX * range);
    }
    printf("\n");
}

void run_random_always_set_time(int size, float range) {
    for (int i = 0; i < size; i++) {
        srand((int)time(0));
        printf("i:%d rand:%d\n", i, rand());
    }
    printf("\n");
    
    for (int i = 0; i < size; i++) {
        srand((int)time(0));
        printf("i:%d rand:%f\n", i, rand() * 1.0 / RAND_MAX);
    }
    printf("\n");

    for (int i = 0; i < size; i++) {
        srand((int)time(0));
        printf("i:%d rand:%f\n", i, rand() * 1.0 / RAND_MAX * range);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    if (3 != argc) {
        fprintf(stderr, "Usage: %s random_num random_range\n");
        return -1;
    }
    
    int size = atoi(argv[1]); // 5
    float range = atof(argv[2]); // 10, 0.1, ...

    printf("srand seed use default: 1\nrandom value is all same, even if retry compiling or retry running\n");
    run_random(size, range);
    
    printf("srand seed use: 1\nrandom value is all same, even if retry compiling or retry running\n");
    srand(1);
    run_random(size, range);
    
    printf("srand seed time(0)\nrandom value is not same when when retry running after 1 sec\nrandom value is all same when when retry running in 1 sec\n");
    srand((int)time(0));
    printf("time(0):%d\n", time(0));
    run_random(size, range);

    printf("srand seed at each time\nrandom value is all same\n");
    run_random_always_set_time(size, range);

    printf("srand seed m_sec time\nrandom value is not same when when retry running after 1 m_sec\nrandom value is all same when when retry running in 1 m_sec\n");
    set_m_sec_seed();
    run_random(size, range);

    return 0;
}
