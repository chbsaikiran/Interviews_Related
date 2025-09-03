#include <stdio.h>
#include <stdlib.h>

// Node structure
struct Node {
    int data;
    struct Node* next;
};

// Function to create a new node
struct Node* createNode(int data) {
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

// Function to detect cycle in linked list
int hasCycle(struct Node* head) {
    struct Node* slow = head;
    struct Node* fast = head;

    while (fast != NULL && fast->next != NULL) {
        slow = slow->next;
        fast = fast->next->next;

        if (slow == fast) {
            return 1; // cycle detected
        }
    }
    return 0; // no cycle
}

// Optional: find start of loop
struct Node* findCycleStart(struct Node* head) {
    struct Node* slow = head;
    struct Node* fast = head;

    while (fast != NULL && fast->next != NULL) {
        slow = slow->next;
        fast = fast->next->next;

        if (slow == fast) {
            slow = head;
            while (slow != fast) {
                slow = slow->next;
                fast = fast->next;
            }
            return slow; // start of cycle
        }
    }
    return NULL; // no cycle
}

int main() {
    struct Node* head = createNode(1);
    head->next = createNode(2);
    head->next->next = createNode(3);
    head->next->next->next = createNode(4);
    head->next->next->next->next = head->next; // create a cycle

    if (hasCycle(head))
        printf("Linked list has a cycle.\n");
    else
        printf("Linked list does not have a cycle.\n");

    struct Node* start = findCycleStart(head);
    if (start)
        printf("Cycle starts at node with value: %d\n", start->data);

    return 0;
}
