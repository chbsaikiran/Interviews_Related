#include <stdio.h>
#include <stdlib.h>

#define MAX_NODES 10

// Queue structure for BFS
int queue[MAX_NODES];
int front = -1, rear = -1;

// Graph represented as adjacency matrix
int graph[MAX_NODES][MAX_NODES];
int visited[MAX_NODES];

// Enqueue operation
void enqueue(int node) {
    if (rear == MAX_NODES - 1)
        return;
    if (front == -1)
        front = 0;
    queue[++rear] = node;
}

// Dequeue operation
int dequeue() {
    if (front == -1 || front > rear)
        return -1;
    return queue[front++];
}

// BFS algorithm
void bfs(int start, int nodes) {
    visited[start] = 1;
    enqueue(start);

    while (front <= rear) {
        int current = dequeue();
        printf("Visited Node: %d\n", current);

        for (int i = 0; i < nodes; i++) {
            if (graph[current][i] && !visited[i]) {
                visited[i] = 1;
                enqueue(i);
            }
        }
    }
}

int main() {
    int nodes = 5;

    // Example graph
    int exampleGraph[5][5] = {
        {0, 1, 1, 0, 0},
        {1, 0, 0, 1, 0},
        {1, 0, 0, 1, 1},
        {0, 1, 1, 0, 1},
        {0, 0, 1, 1, 0}
    };

    // Copy example graph to global graph
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < nodes; j++) {
            graph[i][j] = exampleGraph[i][j];
        }
        visited[i] = 0;
    }

    printf("Breadth-First Search starting from node 0:\n");
    bfs(0, nodes);

    return 0;
}
