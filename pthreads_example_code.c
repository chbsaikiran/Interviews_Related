#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

// Function executed by each thread
void* print_numbers(void* arg) {
    int thread_id = *(int*)arg;

    // Print numbers from 1 to 5 along with thread ID
    for (int i = 1; i <= 5; i++) {
        printf("Thread %d: %d\n", thread_id, i);
    }

    pthread_exit(NULL);
}

int main() {
    int num_threads = 4; // Number of threads
    pthread_t threads[num_threads];
    int thread_args[num_threads];

    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_args[i] = i + 1;
        int ret = pthread_create(&threads[i], NULL, print_numbers, (void*)&thread_args[i]);

        if (ret != 0) {
            printf("Error creating thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("All threads finished.\n");
    return 0;
}
